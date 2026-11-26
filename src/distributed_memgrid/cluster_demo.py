"""End-to-end synthetic coordinator and worker shard assignment demonstration."""

from __future__ import annotations

from distributed_memgrid.protocol import WorkerInfo
from distributed_memgrid.registry import WorkerRegistry
from distributed_memgrid.sharding import (
    Tensor,
    join_shards,
    matrix_multiply,
    split_tensor,
)
from distributed_memgrid.worker import Worker


def main() -> None:
    inputs = Tensor((2, 3), (1.0, 2.0, 3.0, 2.0, 1.0, 0.0))
    weight = Tensor(
        (3, 4),
        (1.0, 0.0, 2.0, 1.0, 0.0, 1.0, 1.0, 2.0, 2.0, 1.0, 0.0, 1.0),
    )
    worker_nodes = [
        Worker("worker-01", "127.0.0.1:50051"),
        Worker("worker-02", "127.0.0.1:50052"),
    ]
    registry = WorkerRegistry()
    for worker in worker_nodes:
        registry.register(WorkerInfo(worker.worker_id, worker.address, memory_bytes=1 << 30))

    available = {info.worker_id for info in registry.list_workers()}
    if available != {worker.worker_id for worker in worker_nodes}:
        raise RuntimeError("coordinator registry does not contain all workers")

    weight_shards = split_tensor(weight, tensor_name="demo.linear.weight", axis=1, shard_count=2)
    output_shards: list[TensorShard] = []
    for worker, shard in zip(worker_nodes, weight_shards, strict=True):
        worker.load_tensor_shard(shard)
        retained = worker.tensor_shards[("demo", "demo.linear.weight")]
        output_shards.append(
            worker.compute_column_shard(
                inputs, tensor_name="demo.linear.weight", model_id="demo"
            )
        )
        print(
            f"{worker.worker_id} at {worker.address}: "
            f"loaded shard {shard.shard_index}/{shard.shard_count}, shape={retained.tensor.shape}"
        )

    distributed_output = join_shards(output_shards)
    reference_output = matrix_multiply(inputs, weight)
    print(f"Coordinator workers: {len(registry.list_workers())}")
    print(f"Gathered worker result: {distributed_output.values}")
    print(f"Single-process reference: {reference_output.values}")
    if distributed_output != reference_output:
        raise RuntimeError("worker result does not match the single-process reference")
    print("PASS: assigned worker shards reproduce the single-process result")


if __name__ == "__main__":
    main()
