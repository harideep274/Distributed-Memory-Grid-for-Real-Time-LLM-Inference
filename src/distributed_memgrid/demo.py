"""Runnable model-parallel tensor sharding demonstration."""

from __future__ import annotations

from distributed_memgrid.sharding import Tensor, column_parallel_linear, matrix_multiply, split_tensor


def main() -> None:
    # Synthetic weights keep this demo quick and independent of external downloads.
    inputs = Tensor((2, 3), (1.0, 2.0, 3.0, 2.0, 1.0, 0.0))
    weight = Tensor(
        (3, 4),
        (
            1.0,
            0.0,
            2.0,
            1.0,
            0.0,
            1.0,
            1.0,
            2.0,
            2.0,
            1.0,
            0.0,
            1.0,
        ),
    )
    bias = Tensor((4,), (0.5, -1.0, 1.5, 0.0))

    shards = split_tensor(weight, tensor_name="demo.linear.weight", axis=1, shard_count=2)
    print("Column-parallel linear demo")
    print(f"Input shape: {inputs.shape}; weight shape: {weight.shape}; workers: {len(shards)}")
    for shard in shards:
        print(
            f"worker-{shard.shard_index}: weight shape={shard.tensor.shape}, "
            f"values={shard.tensor.values}"
        )

    distributed_output = column_parallel_linear(inputs, weight, shard_count=2, bias=bias)
    raw_reference = matrix_multiply(inputs, weight)
    reference_output = Tensor(
        raw_reference.shape,
        tuple(
            value + bias.values[index % weight.shape[1]]
            for index, value in enumerate(raw_reference.values)
        ),
        raw_reference.dtype,
    )
    print(f"Bias: {bias.values}")
    print(f"Gathered output: {distributed_output.values}")
    print(f"Single-process reference: {reference_output.values}")

    if distributed_output != reference_output:
        raise RuntimeError("distributed output does not match the single-process reference")
    print("PASS: gathered worker outputs match the single-process result")


if __name__ == "__main__":
    main()
