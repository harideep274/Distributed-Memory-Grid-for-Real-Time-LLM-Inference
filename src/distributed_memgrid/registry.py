"""In-memory worker registry used by the coordinator API."""

from distributed_memgrid.protocol import WorkerInfo


class WorkerRegistry:
    def __init__(self) -> None:
        self._workers: dict[str, WorkerInfo] = {}

    def register(self, worker: WorkerInfo) -> None:
        if not worker.worker_id.strip():
            raise ValueError("worker_id cannot be empty")
        self._workers[worker.worker_id] = worker

    def get_worker(self, worker_id: str) -> WorkerInfo | None:
        """Return a registered worker, or None if it is not registered."""
        return self._workers.get(worker_id)

    def remove(self, worker_id: str) -> bool:
        return self._workers.pop(worker_id, None) is not None

    def list_workers(self) -> list[WorkerInfo]:
        return sorted(self._workers.values(), key=lambda worker: worker.worker_id)

