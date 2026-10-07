# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Top-level RolloutWorker abstractions (Service vs Client Driver)."""

from concurrent import futures
import dataclasses
import threading
from typing import Any, AsyncIterator, Callable, List, Mapping, Optional, Sequence, Union

from absl import logging
import numpy as np
from tunix.experimental.common import datatypes
from tunix.experimental.common import gcs_cache
from tunix.experimental.rollout import manager as manager_lib
from tunix.experimental.rollout import sampler as sampler_lib
from tunix.experimental.trajectory import store as trajectory_store_lib
from tunix.experimental.trajectory import trajectory as trajectory_lib
from tunix.experimental.weight_sync import weight_sync
from tunix.experimental.worker import abstract_worker
from tunix.rl.rollout import base_rollout


@dataclasses.dataclass
class RolloutConfig(base_rollout.RolloutConfig):
  """Rollout configuration extending base RolloutConfig with sampler choice and registry options.

  Attributes:
    sampler_type: Type of sampler adapter to construct ("vanilla",
      "inprocess_vllm", "vllm").
    weight_sync_mode: Mode of weight synchronization ("none", "fallback",
      "raiden").
    env_name: Registered name of environment class in ENV_REGISTRY.
    agent_name: Registered name of agent class in AGENT_REGISTRY.
    env_config: Configuration dictionary passed to environment constructor.
    agent_config: Configuration dictionary passed to agent constructor.
    trajectory_store_config: Trajectory Store configuration for this worker
      process, or None to run without a store. See
      `store.TrajectoryStore.from_config`. Must match what the orchestrator
      was given: for the file backend it is the shared root_dir and run_id
      that will make these writes visible to the orchestrator's reads once
      rollout step logging is wired.
    partial_rollout: Whether to freeze in-flight trajectories in-place during
      weight synchronization instead of draining them to completion first.
  """

  sampler_type: str = "vanilla"
  weight_sync_mode: weight_sync.WeightSyncMode = weight_sync.WeightSyncMode.NONE
  env_name: str = ""
  agent_name: str = ""
  env_config: dict[str, Any] = dataclasses.field(default_factory=dict)
  agent_config: dict[str, Any] = dataclasses.field(default_factory=dict)
  trajectory_store_config: Mapping[str, Any] | None = None
  partial_rollout: bool = False


TrajectoryOrError = Union[
    trajectory_lib.Trajectory, trajectory_lib.TrajectoryError
]

WorkerState = datatypes.WorkerState


class RolloutWorker(abstract_worker.Worker):
  """Worker wrapper for rollout collection.

  Encapsulates RolloutManager and executes concurrent episode loops
  locally on its remote CPU host.
  """

  def __init__(
      self,
      worker_id: str,
      config: Optional[RolloutConfig] = None,
      sampler: Optional[sampler_lib.Sampler] = None,
      env_pool: Any = None,
      agent_factory: Optional[Callable[[], Any]] = None,
      max_concurrency: int = 64,
      tokenizer: Any = None,
      chat_parser: Any = None,
      jax_cache_config: Optional[gcs_cache.JaxCacheConfig] = None,
  ):
    super().__init__()
    self.worker_id = worker_id
    self.config = config
    self._policy_version = 0
    self._state = datatypes.WorkerState.PENDING
    self._init_lock = threading.Lock()
    self._sync_round = {"req_id": None, "uuid": 0, "phase": "idle"}
    self.jax_cache_config: gcs_cache.JaxCacheConfig = (
        jax_cache_config
        if jax_cache_config is not None
        else gcs_cache.JaxCacheConfig.from_env()
    )
    self._pending_jax_cache_sync: tuple[futures.Future[bool], float] | None = (
        None
    )
    self._first_rollout_cache_synced: bool = False
    self._jax_cache_lock = threading.Lock()
    if tokenizer is None or chat_parser is None:
      raise ValueError(
          "RolloutWorker requires valid tokenizer and chat_parser arguments"
          " (none can be None)."
      )
    self.manager = manager_lib.RolloutManager(
        config=config,
        sampler=sampler,
        env_pool=env_pool,
        agent_factory=agent_factory,
        max_concurrency=max_concurrency,
        tokenizer=tokenizer,
        chat_parser=chat_parser,
    )
    # Built at most once per process: this __init__ runs exactly once per
    # RolloutWorker instance, so there is no separate guard against
    # constructing the store twice. See store.TrajectoryStore.from_config.
    # TODO(sizhi): Pass self._trajectory_store into RolloutManager / collector
    # to log rollout steps in follow-up CLs.
    store_config = (
        {
            trajectory_store_lib.METADATA_TYPE_KEY: (
                trajectory_lib.TunixTrajectoryMetadata.METADATA_TYPE
            ),
            **config.trajectory_store_config,
        }
        if config is not None and config.trajectory_store_config is not None
        else None
    )
    self._trajectory_store = trajectory_store_lib.TrajectoryStore.from_config(
        store_config
    )
    if self._trajectory_store is not None:
      # Several workers can share one log stream, and absl log lines carry no
      # process identity, so the worker_id is what attributes a reported
      # config to a process.
      logging.info(
          "[trajectory-store] worker %s built %s",
          worker_id,
          self._trajectory_store.to_redacted_config(),
      )

  @property
  def trajectory_store(self) -> trajectory_store_lib.TrajectoryStore | None:
    return self._trajectory_store

  @property
  def sampler(self) -> sampler_lib.Sampler:
    return self.manager.sampler

  def get_worker_id(self) -> str:
    """Returns the unique worker ID."""
    return self.worker_id

  def info(self) -> datatypes.WorkerInfo:
    return datatypes.WorkerInfo(
        worker_id=self.worker_id,
        roles=frozenset({"rollout"}),
        resources={
            "sampler": type(self.sampler).__name__,
            "policy_version": self._policy_version,
        },
    )

  def _response(self, **metadata: Any) -> datatypes.Response:
    return datatypes.Response(
        metadata={
            "worker_id": self.worker_id,
            "state": self.state.value,
            "policy_version": self._policy_version,
            **metadata,
        }
    )

  def _resolve_jax_cache_gcs_uri(self) -> str | None:
    """Resolves the target rollout GCS compilation cache URI, if configured."""
    return (
        self.jax_cache_config.resolved_gcs_uri or gcs_cache.get_active_gcs_uri()
    )

  def _await_jax_cache_upload(
      self, pending: tuple[futures.Future[bool], float]
  ) -> None:
    """Waits for a specific JAX cache upload future and logs its outcome."""
    outcome, sync_timeout_s = pending
    try:
      uploaded = outcome.result(timeout=sync_timeout_s)
    except futures.TimeoutError:
      logging.warning(
          "JAX cache upload on rollout worker %s timed out after %.0fs;"
          " abandoning it.",
          self.worker_id,
          sync_timeout_s,
      )
      return
    except Exception as err:  # pylint: disable=broad-except
      logging.warning(
          "Failed to sync JAX cache on rollout worker %s: %r",
          self.worker_id,
          err,
      )
      return
    if not uploaded:
      logging.warning(
          "Rollout worker %s reported a failed JAX cache upload.",
          self.worker_id,
      )
      return
    logging.info("Rollout worker %s JAX cache upload finished.", self.worker_id)

  def _wait_for_jax_cache_sync(self) -> None:
    """Waits for any in-flight background JAX cache upload to finish."""
    with self._jax_cache_lock:
      pending = self._pending_jax_cache_sync
      self._pending_jax_cache_sync = None
    if pending is not None:
      self._await_jax_cache_upload(pending)

  def sync_jax_cache(
      self, *, wait: bool = False
  ) -> futures.Future[bool] | None:
    """Autonomously synchronizes local JAX compilation cache to GCS on the primary replica.

    Args:
      wait: If True, blocks until the upload completes (or times out). If False,
        dispatches the upload on a background daemon thread and returns its
        Future immediately so rollout critical paths are not blocked.

    Returns:
      The upload Future when an upload is launched, or None if cache persistence
      is disabled or this worker is a non-primary replica.
    """
    if (
        not self.jax_cache_config.save_jax_cache
        or gcs_cache.is_jax_cache_disabled()
        or not gcs_cache.is_primary_rollout_worker(self.worker_id)
    ):
      return None
    gcs_uri = self._resolve_jax_cache_gcs_uri()
    if not gcs_uri:
      return None

    local_dir = self.jax_cache_config.local_dir
    sync_timeout_s = self.jax_cache_config.sync_timeout_s
    logging.info(
        "Triggering autonomous JAX compilation cache upload to GCS (%s) on"
        " rollout worker %s (wait=%s)...",
        gcs_uri,
        self.worker_id,
        wait,
    )

    outcome: futures.Future[bool] = futures.Future()
    with self._jax_cache_lock:
      prior_pending = self._pending_jax_cache_sync
      self._pending_jax_cache_sync = (outcome, sync_timeout_s)

    def _run() -> None:
      if prior_pending is not None:
        self._await_jax_cache_upload(prior_pending)
      try:
        outcome.set_result(
            gcs_cache.save_jax_cache(gcs_uri=gcs_uri, local_dir=local_dir)
        )
      except Exception as err:  # pylint: disable=broad-except
        outcome.set_exception(err)

    threading.Thread(
        target=_run,
        name=f"jax-cache-upload-{self.worker_id}",
        daemon=True,
    ).start()
    if wait:
      self._wait_for_jax_cache_sync()
    return outcome

  def _maybe_sync_jax_cache_after_first_rollout(
      self, response: datatypes.RolloutResponse
  ) -> None:
    """Triggers a one-time background cache upload after the first completed rollout."""
    if response.status != "COMPLETED" or self._first_rollout_cache_synced:
      return
    with self._jax_cache_lock:
      if self._first_rollout_cache_synced:
        return
      self._first_rollout_cache_synced = True
    self.sync_jax_cache(wait=False)

  def _ensure_initialized(self) -> None:
    if self.state == WorkerState.PENDING:
      self.initialize()

  def initialize(self) -> datatypes.Response:
    with self._init_lock:
      if self.state == WorkerState.READY:
        return self._response(initialized=True, ready=True)
      self.state = WorkerState.INITIALIZING
      try:
        self.sampler.initialize()
        return self._response()
      except Exception:
        self.state = WorkerState.ERROR
        raise
      finally:
        if self.state == WorkerState.INITIALIZING:
          self.state = WorkerState.READY

  def compile(self, dummy_data: Any) -> datatypes.Response:
    self._ensure_initialized()
    self.state = WorkerState.COMPILING
    try:
      return datatypes.Response()
    finally:
      self.state = WorkerState.READY

  async def start(self) -> datatypes.Response:
    self._ensure_initialized()
    try:
      await self.sampler.start()
      await self.manager.bind_weight_sync()
      self.sync_jax_cache(wait=False)
      return self._response(started=True)
    except Exception:
      self.state = WorkerState.ERROR
      raise

  def stop(self) -> datatypes.Response:
    self.state = WorkerState.STOPPED
    try:
      self.manager.cancel_all()
    finally:
      try:
        # Runs even when cancel_all raises, so a failed stop releases the
        # store's background writer thread instead of leaking it.
        if self._trajectory_store is not None:
          self._trajectory_store.close()
      finally:
        self._wait_for_jax_cache_sync()
    return datatypes.Response()

  def pause(self) -> datatypes.Response:
    self.manager.pause_all()
    return datatypes.Response()

  def resume(self) -> datatypes.Response:
    self.manager.resume_all()
    return datatypes.Response()

  def _infer_shapes(self) -> Any:
    return None

  def _compile_with_shapes(self, abstract_state: Any) -> None:
    pass

  def heartbeat(self) -> datatypes.HealthReport:
    return datatypes.HealthReport(
        state=self.state,
        policy_version=self._policy_version,
        inflight=len(self.manager._active_tasks),  # pylint: disable=protected-access
        queue_depth=self.manager._completed_queue.qsize(),  # pylint: disable=protected-access
    )

  def get_target_state(self) -> Any:
    """Returns rollout-side target-state skeleton for trainer-side conversion."""
    self._ensure_initialized()
    return self.manager.get_target_state()

  def _stamp_worker_lineage(self, metadata: dict[str, Any] | None) -> None:
    """Appends worker generation telemetry to the lineage context if present."""
    if metadata is None:
      return
    lineage_ctx = metadata.get("lineage")
    if lineage_ctx is not None and hasattr(lineage_ctx, "add_event"):
      lineage_ctx.add_event(
          component="worker.rollout",
          operation="generate",
          attributes={"worker_id": self.worker_id},
      )

  def _to_rollout_response(
      self,
      item: Any,
      request_id: str = "",
      prompt_tokens: np.ndarray | None = None,
  ) -> datatypes.RolloutResponse:
    """Converts internal Trajectory or TrajectoryError to wire-safe RolloutResponse."""
    if isinstance(item, trajectory_lib.TrajectoryError):
      metadata = dict(item.metadata or {})
      metadata["prompt_id"] = item.prompt_id
      self._stamp_worker_lineage(metadata)
      return datatypes.RolloutResponse(
          request_id=request_id or item.trajectory_id or item.prompt_id,
          status="ERROR",
          error=datatypes.ErrorInfo(
              error_type="TrajectoryError",
              message=str(item.error_message),
          ),
          payload=None,
          metadata=metadata,
      )
    if isinstance(item, datatypes.RolloutResponse):
      return item
    if isinstance(item, datatypes.TrajectoryItem):
      req_id = request_id or getattr(item, "traj_id", "")
      if prompt_tokens is not None and getattr(item, "prompt_tokens", None) is None:
        item.metadata["prompt_tokens"] = prompt_tokens
      self._stamp_worker_lineage(item.metadata)
      return datatypes.RolloutResponse(
          request_id=req_id,
          status="COMPLETED",
          payload=item,
          metadata=item.metadata,
      )
    raise TypeError(
        f"Unsupported item type for RolloutResponse conversion: {type(item)}"
    )

  async def generate(
      self,
      requests: datatypes.RolloutRequest | Sequence[datatypes.RolloutRequest],
      on_complete: Optional[Callable[[datatypes.RolloutResponse], None]] = None,
  ) -> datatypes.RolloutResponse | List[datatypes.RolloutResponse]:
    """Coroutine method for single or batched generate requests."""
    self._ensure_initialized()
    if isinstance(requests, datatypes.RolloutRequest):
      pass
    elif isinstance(requests, Sequence) and not isinstance(
        requests, (str, bytes)
    ):
      if not all(isinstance(req, datatypes.RolloutRequest) for req in requests):
        raise TypeError(
            "generate requires `requests` to be a RolloutRequest or"
            " Sequence[RolloutRequest]."
        )
    else:
      raise TypeError(
          "generate requires `requests` to be a RolloutRequest or"
          " Sequence[RolloutRequest]."
      )

    cb = None
    if on_complete is not None:
      cb = lambda item: on_complete(self._to_rollout_response(item))
    res = await self.manager.generate(requests, on_complete=cb)
    if isinstance(res, (list, tuple)):
      responses = [self._to_rollout_response(r) for r in res]
      for resp in responses:
        self._maybe_sync_jax_cache_after_first_rollout(resp)
      return responses
    resp = self._to_rollout_response(res)
    self._maybe_sync_jax_cache_after_first_rollout(resp)
    return resp

  async def pop_next_completed(self) -> datatypes.RolloutResponse | Any:
    """Pull-based stream: yields whichever trajectory finishes first out-of-order."""
    res = await self.manager.pop_next_completed()
    resp = self._to_rollout_response(res)
    self._maybe_sync_jax_cache_after_first_rollout(resp)
    return resp

  async def as_completed_stream(
      self,
  ) -> AsyncIterator[datatypes.RolloutResponse | Any]:
    """Async stream yielding completed trajectories or errors strictly out-of-order."""
    async for res in self.manager.as_completed_stream():
      resp = self._to_rollout_response(res)
      self._maybe_sync_jax_cache_after_first_rollout(resp)
      yield resp

  async def pre_weight_sync(self, sync_request: Any = None, **kwargs) -> Any:
    """Quiesces the worker; it stays SYNCING until post or abort."""
    self._ensure_initialized()
    self.state = WorkerState.SYNCING
    self._record_round(sync_request, "idle")
    result = await self.manager.pre_weight_sync(sync_request, **kwargs)
    self._record_round(sync_request, "prepared")
    return result

  async def weight_sync(self, sync_request: Any = None, **kwargs) -> Any:
    """Materializes the received weights; the worker stays SYNCING."""
    self._ensure_initialized()
    self.state = WorkerState.SYNCING
    metadata = kwargs.pop("metadata", None)
    request = sync_request if sync_request is not None else metadata
    result = await self.manager.weight_sync(request, **kwargs)
    if isinstance(result, int):
      self._policy_version = result
    else:
      version = getattr(request, "policy_version", None)
      if version is not None:
        self._policy_version = version
      else:
        self._policy_version += 1
    self._record_round(request, "h2d_done")
    return result

  async def post_weight_sync(self, sync_request: Any = None, **kwargs) -> Any:
    """Publishes the new weights and resumes serving."""
    result = await self.manager.post_weight_sync(sync_request, **kwargs)
    self.state = WorkerState.READY
    self._record_round(sync_request, "committed")
    return result

  async def bind_weight_sync(self, **kwargs) -> Any:
    """Binds the destination-side transport via the manager."""
    self._ensure_initialized()
    return await self.manager.bind_weight_sync(**kwargs)

  async def get_weight_sync_metadata(self, **kwargs) -> Any:
    """Returns the sampler's transport metadata via the manager."""
    self._ensure_initialized()
    return await self.manager.get_weight_sync_metadata(**kwargs)

  async def abort_weight_sync(self, sync_request: Any = None, **kwargs) -> Any:
    """Discards the round and resumes serving the previous weights."""
    res = await self.manager.abort_weight_sync(sync_request, **kwargs)
    self.state = WorkerState.READY
    self._record_round(sync_request, "aborted")
    return res

  async def get_weight_sync_status(self, **kwargs) -> Any:
    """Returns this worker's view of the current weight sync round."""
    return dict(self._sync_round, policy_version=self._policy_version)

  def _record_round(self, sync_request: Any, phase: str) -> None:
    extra = getattr(sync_request, "extra_config", None) or {}
    if extra.get("req_id") is not None:
      self._sync_round["req_id"] = extra.get("req_id")
      self._sync_round["uuid"] = extra.get("uuid", 0)
    self._sync_round["phase"] = phase
