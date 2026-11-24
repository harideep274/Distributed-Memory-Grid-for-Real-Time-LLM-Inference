"""Coordinator API for worker registration and cluster inspection."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from distributed_memgrid.protocol import WorkerInfo, WorkerState
from distributed_memgrid.registry import WorkerRegistry

app = FastAPI(title="Distributed Memory Grid Master", version="0.1.0")
registry = WorkerRegistry()


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/workers")
async def list_workers() -> list[WorkerInfo]:
    return registry.list_workers()


class WorkerRegistration(BaseModel):
    worker_id: str = Field(min_length=1, max_length=128)
    address: str = Field(min_length=1, max_length=512)
    memory_bytes: int = Field(default=0, ge=0)
    labels: dict[str, str] = Field(default_factory=dict)


class WorkerHeartbeat(BaseModel):
    state: WorkerState = WorkerState.READY


@app.post("/workers", status_code=201)
async def register_worker(payload: WorkerRegistration) -> WorkerInfo:
    worker = WorkerInfo(
        worker_id=payload.worker_id.strip(),
        address=payload.address,
        memory_bytes=payload.memory_bytes,
        labels=payload.labels,
    )
    if not worker.worker_id:
        raise HTTPException(status_code=422, detail="worker_id cannot be blank")
    registry.register(worker)
    return worker


@app.post("/workers/{worker_id}/heartbeat")
async def worker_heartbeat(worker_id: str, payload: WorkerHeartbeat) -> WorkerInfo:
    worker = registry.heartbeat(worker_id, payload.state)
    if worker is None:
        raise HTTPException(status_code=404, detail="worker is not registered")
    return worker


@app.get("/cluster")
async def cluster_summary() -> dict[str, int | str]:
    workers = registry.list_workers()
    return {
        "status": "ok",
        "worker_count": len(workers),
        "ready_workers": sum(worker.state == WorkerState.READY for worker in workers),
        "reported_memory_bytes": sum(worker.memory_bytes for worker in workers),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("distributed_memgrid.master:app", host="127.0.0.1", port=8000, reload=False)
