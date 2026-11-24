"""Runnable demonstration of row-parallel linear computation."""

from __future__ import annotations

from distributed_memgrid.sharding import (
    Tensor,
    matrix_multiply,
    row_parallel_linear,
    split_tensor,
)


def main() -> None:
    # Synthetic tensors make the demo quick and independent of model downloads.
    inputs = Tensor((2, 4), (1.0, 2.0, 3.0, 4.0, 2.0, 1.0, 0.0, 1.0))
    weight = Tensor(
        (4, 2),
        (
            1.0,
            0.0,
            0.0,
            1.0,
            2.0,
            1.0,
            1.0,
            2.0,
        ),
    )
    bias = Tensor((2,), (0.5, -1.0))

    input_shards = split_tensor(
        inputs, tensor_name="demo.linear.input", axis=1, shard_count=2
    )
    weight_shards = split_tensor(
        weight, tensor_name="demo.linear.weight", axis=0, shard_count=2
    )

    print("Row-parallel linear demo")
    print(f"Input shape: {inputs.shape}; weight shape: {weight.shape}; workers: 2")
    for input_shard, weight_shard in zip(input_shards, weight_shards, strict=True):
        partial = matrix_multiply(input_shard.tensor, weight_shard.tensor)
        print(
            f"worker-{input_shard.shard_index}: input shape={input_shard.tensor.shape}, "
            f"weight shape={weight_shard.tensor.shape}, partial={partial.values}"
        )

    reduced_output = row_parallel_linear(inputs, weight, shard_count=2, bias=bias)
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
    print(f"Reduced worker outputs: {reduced_output.values}")
    print(f"Single-process reference: {reference_output.values}")

    if reduced_output != reference_output:
        raise RuntimeError("reduced worker outputs do not match the single-process reference")
    print("PASS: reduced worker outputs match the single-process result")


if __name__ == "__main__":
    main()
