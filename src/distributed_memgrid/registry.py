"""In-memory worker registry used by the coordinator API."""

from dataclasses import replace
from time import monotonic

from distributed_memgrid.protocol import WorkerInfo, WorkerState


class WorkerRegistry:
    def __init__(self) -> None:
        self._workers: dict[str, WorkerInfo] = {}
        self._last_heartbeat: dict[str, float] = {}

    def register(self, worker: WorkerInfo) -> None:
        if not worker.worker_id.strip():
            raise ValueError("worker_id cannot be empty")
        self._workers[worker.worker_id] = worker
        self._last_heartbeat[worker.worker_id] = monotonic()

    def heartbeat(
        self, worker_id: str, state: WorkerState = WorkerState.READY
    ) -> WorkerInfo | None:
        """Record a worker heartbeat and its current state; return None if unknown."""
        worker = self._workers.get(worker_id)
        if worker is None:
            return None
        updated = replace(worker, state=state)
        self._workers[worker_id] = updated
        self._last_heartbeat[worker_id] = monotonic()
        return updated

    def mark_stale(self, timeout_seconds: float) -> None:
        """Mark workers unavailable when their last heartbeat exceeds the timeout."""
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        cutoff = monotonic() - timeout_seconds
        for worker_id, last_seen in self._last_heartbeat.items():
            worker = self._workers[worker_id]
            if last_seen < cutoff and worker.state != WorkerState.UNAVAILABLE:
                self._workers[worker_id] = replace(worker, state=WorkerState.UNAVAILABLE)

    def get_worker(self, worker_id: str) -> WorkerInfo | None:
        """Return a registered worker, or None if it is not registered."""
        return self._workers.get(worker_id)

    def remove(self, worker_id: str) -> bool:
        removed = self._workers.pop(worker_id, None) is not None
        self._last_heartbeat.pop(worker_id, None)
        return removed

    def list_workers(self) -> list[WorkerInfo]:
        self.mark_stale(30.0)
        return sorted(self._workers.values(), key=lambda worker: worker.worker_id)

