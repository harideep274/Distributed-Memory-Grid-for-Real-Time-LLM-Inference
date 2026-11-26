"""In-process worker boundary for validating and holding assigned tensor shards."""

from distributed_memgrid.protocol import WeightShard
from distributed_memgrid.sharding import Tensor, TensorShard, matrix_multiply


class Worker:
    def __init__(self, worker_id: str, address: str) -> None:
        self.worker_id = worker_id
        self.address = address
        self.shards: list[WeightShard] = []
        self.tensor_shards: dict[tuple[str, str], TensorShard] = {}

    def assign_shard(self, shard: WeightShard) -> None:
        """Record a validated metadata-only assignment."""
        if shard.shard_count < 1 or not 0 <= shard.shard_index < shard.shard_count:
            raise ValueError("shard_index must be in [0, shard_count)")
        if not shard.model_id.strip() or not shard.tensor_name.strip():
            raise ValueError("model_id and tensor_name cannot be empty")
        if not shard.shape or any(dimension <= 0 for dimension in shard.shape):
            raise ValueError("shard shape must contain positive dimensions")
        if not shard.dtype:
            raise ValueError("dtype cannot be empty")
        self.shards.append(shard)

    def load_tensor_shard(self, shard: TensorShard, *, model_id: str = "demo") -> None:
        """Validate a synthetic tensor shard and retain it under its model name.

        This models the worker-side boundary for the demo. It does not load a
        checkpoint or transfer data over a network.
        """
        if not model_id.strip() or not shard.tensor_name.strip():
            raise ValueError("model_id and tensor_name cannot be empty")
        if shard.shard_count < 1 or not 0 <= shard.shard_index < shard.shard_count:
            raise ValueError("shard_index must be in [0, shard_count)")
        if not 0 <= shard.axis < len(shard.source_shape):
            raise ValueError("shard axis is out of range")
        if len(shard.tensor.shape) != len(shard.source_shape):
            raise ValueError("shard rank does not match source tensor rank")
        if any(dimension <= 0 for dimension in shard.source_shape):
            raise ValueError("source dimensions must be positive")
        source_width = shard.source_shape[shard.axis]
        if source_width % shard.shard_count:
            raise ValueError("source axis size must be divisible by shard count")
        expected_shape = list(shard.source_shape)
        expected_shape[shard.axis] //= shard.shard_count
        if tuple(expected_shape) != shard.tensor.shape:
            raise ValueError("shard tensor shape does not match assignment metadata")

        key = (model_id, shard.tensor_name)
        previous = self.tensor_shards.get(key)
        if previous is not None:
            if previous != shard:
                raise ValueError("worker already holds a different shard for this tensor")
            return
        self.tensor_shards[key] = shard

    def compute_column_shard(
        self,
        inputs: Tensor,
        *,
        tensor_name: str,
        model_id: str = "demo",
    ) -> TensorShard:
        """Compute this worker's local output for a column-sharded linear layer."""
        try:
            weight_shard = self.tensor_shards[(model_id, tensor_name)]
        except KeyError as exc:
            raise KeyError(f"worker has no shard for {model_id}/{tensor_name}") from exc
        if weight_shard.axis != 1 or len(weight_shard.tensor.shape) != 2:
            raise ValueError("column-parallel compute requires a rank-2 shard on axis 1")
        if len(inputs.shape) != 2 or inputs.shape[1] != weight_shard.tensor.shape[0]:
            raise ValueError("input features do not match the worker's weight shard")

        local_output = matrix_multiply(inputs, weight_shard.tensor)
        return TensorShard(
            tensor_name=f"{tensor_name}.output",
            shard_index=weight_shard.shard_index,
            shard_count=weight_shard.shard_count,
            axis=1,
            source_shape=(inputs.shape[0], weight_shard.source_shape[1]),
            tensor=local_output,
        )
