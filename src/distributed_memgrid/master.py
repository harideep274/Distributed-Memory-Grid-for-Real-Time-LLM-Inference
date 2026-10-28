"""Development master process with health and worker-status endpoints."""

from fastapi import FastAPI

from distributed_memgrid.protocol import WorkerInfo
from distributed_memgrid.registry import WorkerRegistry

app = FastAPI(title="Distributed Memory Grid Master", version="0.1.0")
registry = WorkerRegistry()


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/workers")
async def list_workers() -> list[WorkerInfo]:
    return registry.list_workers()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("distributed_memgrid.master:app", host="127.0.0.1", port=8000, reload=False)
