"""Worker process boundary; model loading and gRPC serving are Week 1 tasks."""

from distributed_memgrid.protocol import WeightShard


class Worker:
    def __init__(self, worker_id: str, address: str) -> None:
        self.worker_id = worker_id
        self.address = address
        self.shards: list[WeightShard] = []

    def assign_shard(self, shard: WeightShard) -> None:
        """Record a shard assignment; tensor loading is implemented in the next step."""
        if shard.shard_count < 1 or not 0 <= shard.shard_index < shard.shard_count:
            raise ValueError("shard_index must be in [0, shard_count)")
        self.shards.append(shard)
