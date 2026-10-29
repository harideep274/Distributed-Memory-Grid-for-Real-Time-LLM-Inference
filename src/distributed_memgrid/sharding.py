"""Dependency-free tensor partitioning primitives for the learning demo.

The implementation stores tensor values in row-major order. It is intended to
make layout and correctness rules explicit before adding a framework backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod


@dataclass(frozen=True, slots=True)
class Tensor:
    """A small dense tensor backed by a flat, row-major tuple of numbers."""

    shape: tuple[int, ...]
    values: tuple[float, ...]
    dtype: str = "float32"

    def __post_init__(self) -> None:
        if not self.shape:
            raise ValueError("tensor shape must have at least one dimension")
        if any(dimension <= 0 for dimension in self.shape):
            raise ValueError("tensor dimensions must be positive")
        if len(self.values) != prod(self.shape):
            raise ValueError(
                f"shape {self.shape} requires {prod(self.shape)} values; got {len(self.values)}"
            )
        if not self.dtype:
            raise ValueError("dtype cannot be empty")


@dataclass(frozen=True, slots=True)
class TensorShard:
    """One contiguous partition of a tensor along a selected axis."""

    tensor_name: str
    shard_index: int
    shard_count: int
    axis: int
    source_shape: tuple[int, ...]
    tensor: Tensor


def split_tensor(
    tensor: Tensor,
    *,
    tensor_name: str,
    axis: int,
    shard_count: int,
) -> list[TensorShard]:
    """Split a tensor evenly along ``axis`` using contiguous row-major slices.

    Uneven splits are rejected instead of silently padding model weights. A
    future backend can add padding while retaining this strict default.
    """

    normalized_axis = axis + len(tensor.shape) if axis < 0 else axis
    if not 0 <= normalized_axis < len(tensor.shape):
        raise ValueError(f"axis {axis} is out of range for rank-{len(tensor.shape)} tensor")
    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    axis_size = tensor.shape[normalized_axis]
    if axis_size % shard_count:
        raise ValueError(
            f"dimension {axis_size} on axis {normalized_axis} is not divisible by "
            f"{shard_count} shards"
        )

    shard_width = axis_size // shard_count
    inner_size = prod(tensor.shape[normalized_axis + 1 :])
    outer_size = prod(tensor.shape[:normalized_axis])
    shard_shape = list(tensor.shape)
    shard_shape[normalized_axis] = shard_width
    shards: list[TensorShard] = []

    for shard_index in range(shard_count):
        values: list[float] = []
        start_in_axis = shard_index * shard_width
        for outer_index in range(outer_size):
            block_start = (outer_index * axis_size + start_in_axis) * inner_size
            block_end = block_start + shard_width * inner_size
            values.extend(tensor.values[block_start:block_end])
        shards.append(
            TensorShard(
                tensor_name=tensor_name,
                shard_index=shard_index,
                shard_count=shard_count,
                axis=normalized_axis,
                source_shape=tensor.shape,
                tensor=Tensor(tuple(shard_shape), tuple(values), tensor.dtype),
            )
        )
    return shards


def join_shards(shards: list[TensorShard]) -> Tensor:
    """Reconstruct a tensor from all of its ordered, compatible shards."""

    if not shards:
        raise ValueError("at least one shard is required")
    ordered = sorted(shards, key=lambda shard: shard.shard_index)
    first = ordered[0]
    if len(ordered) != first.shard_count:
        raise ValueError(f"expected {first.shard_count} shards; got {len(ordered)}")

    expected_indices = list(range(first.shard_count))
    if [shard.shard_index for shard in ordered] != expected_indices:
        raise ValueError("shard indices must contain each index exactly once")

    for shard in ordered:
        if (
            shard.tensor_name != first.tensor_name
            or shard.shard_count != first.shard_count
            or shard.axis != first.axis
            or shard.source_shape != first.source_shape
            or shard.tensor.dtype != first.tensor.dtype
            or len(shard.tensor.shape) != len(first.tensor.shape)
        ):
            raise ValueError("shards have incompatible metadata")
        for dimension, (actual, expected) in enumerate(zip(shard.tensor.shape, first.tensor.shape)):
            if dimension != first.axis and actual != expected:
                raise ValueError("shards differ along a non-sharded dimension")
            if dimension == first.axis and actual != first.tensor.shape[dimension]:
                raise ValueError("shards have inconsistent widths")

    full_shape = first.source_shape
    axis_size = full_shape[first.axis]
    shard_width = axis_size // first.shard_count
    if shard_width * first.shard_count != axis_size:
        raise ValueError("source axis size is not divisible by shard count")
    expected_shard_shape = list(full_shape)
    expected_shard_shape[first.axis] = shard_width
    if any(list(shard.tensor.shape) != expected_shard_shape for shard in ordered):
        raise ValueError("shard shape does not match source shape and partition layout")

    inner_size = prod(full_shape[first.axis + 1 :])
    outer_size = prod(full_shape[: first.axis])
    values: list[float] = []
    for outer_index in range(outer_size):
        for shard in ordered:
            start = outer_index * shard_width * inner_size
            end = start + shard_width * inner_size
            values.extend(shard.tensor.values[start:end])
    return Tensor(full_shape, tuple(values), first.tensor.dtype)


def matrix_multiply(left: Tensor, right: Tensor) -> Tensor:
    """Multiply two rank-2 tensors: ``(m, k) @ (k, n) -> (m, n)``."""

    if len(left.shape) != 2 or len(right.shape) != 2:
        raise ValueError("matrix_multiply requires rank-2 tensors")
    rows, inner = left.shape
    right_inner, columns = right.shape
    if inner != right_inner:
        raise ValueError(f"incompatible matrix shapes {left.shape} and {right.shape}")

    output: list[float] = []
    for row in range(rows):
        for column in range(columns):
            total = 0.0
            for index in range(inner):
                total += left.values[row * inner + index] * right.values[index * columns + column]
            output.append(total)
    return Tensor((rows, columns), tuple(output), "float32")


def column_parallel_linear(
    inputs: Tensor,
    weight: Tensor,
    shard_count: int,
    bias: Tensor | None = None,
) -> Tensor:
    """Simulate column-parallel linear: each worker computes output columns.

    ``weight`` uses the common linear-layer layout ``(input_features,
    output_features)``. Each worker receives a slice along axis 1, computes its
    local output, and outputs are gathered along axis 1. An optional bias vector
    is split by output column and added to each worker's local output.
    """

    if len(weight.shape) != 2 or len(inputs.shape) != 2:
        raise ValueError("column_parallel_linear requires rank-2 input and weight tensors")
    if inputs.shape[1] != weight.shape[0]:
        raise ValueError(f"input shape {inputs.shape} does not match weight shape {weight.shape}")
    if bias is not None and bias.shape != (weight.shape[1],):
        raise ValueError(f"bias must have shape ({weight.shape[1]},); got {bias.shape}")

    weight_shards = split_tensor(
        weight, tensor_name="linear.weight", axis=1, shard_count=shard_count
    )
    bias_shards = (
        split_tensor(bias, tensor_name="linear.bias", axis=0, shard_count=shard_count)
        if bias is not None
        else None
    )
    output_shards: list[TensorShard] = []
    for weight_shard in weight_shards:
        local_output = matrix_multiply(inputs, weight_shard.tensor)
        if bias_shards is not None:
            local_bias = bias_shards[weight_shard.shard_index].tensor.values
            local_width = len(local_bias)
            local_output = Tensor(
                local_output.shape,
                tuple(
                    value + local_bias[index % local_width]
                    for index, value in enumerate(local_output.values)
                ),
                local_output.dtype,
            )
        output_shards.append(
            TensorShard(
                tensor_name="linear.output",
                shard_index=weight_shard.shard_index,
                shard_count=weight_shard.shard_count,
                axis=1,
                source_shape=(inputs.shape[0], weight.shape[1]),
                tensor=local_output,
            )
        )
    return join_shards(output_shards)


def row_parallel_linear(
    inputs: Tensor,
    weight: Tensor,
    shard_count: int,
    bias: Tensor | None = None,
) -> Tensor:
    """Simulate row-parallel linear by summing workers' partial outputs.

    ``weight`` uses shape ``(input_features, output_features)``. Both the input
    features and weight rows are split along axis 0/1 respectively. Each worker
    multiplies its matching slices, then the partial outputs are summed, as in
    the reduce step of row-wise tensor parallelism. An optional bias is added
    once after reduction, not once on every worker.
    """

    if len(weight.shape) != 2 or len(inputs.shape) != 2:
        raise ValueError("row_parallel_linear requires rank-2 input and weight tensors")
    if inputs.shape[1] != weight.shape[0]:
        raise ValueError(f"input shape {inputs.shape} does not match weight shape {weight.shape}")
    if bias is not None and bias.shape != (weight.shape[1],):
        raise ValueError(f"bias must have shape ({weight.shape[1]},); got {bias.shape}")

    input_shards = split_tensor(
        inputs, tensor_name="linear.input", axis=1, shard_count=shard_count
    )
    weight_shards = split_tensor(
        weight, tensor_name="linear.weight", axis=0, shard_count=shard_count
    )

    partial_outputs = [
        matrix_multiply(input_shard.tensor, weight_shard.tensor)
        for input_shard, weight_shard in zip(input_shards, weight_shards, strict=True)
    ]
    output_values = [
        sum(partial.values[index] for partial in partial_outputs)
        for index in range(len(partial_outputs[0].values))
    ]
    if bias is not None:
        output_width = weight.shape[1]
        for row in range(inputs.shape[0]):
            for column in range(output_width):
                index = row * output_width + column
                output_values[index] += bias.values[column]
    return Tensor(partial_outputs[0].shape, tuple(output_values), "float32")
