"""Shared transport-independent message types for master and worker components."""

from dataclasses import dataclass, field
from enum import StrEnum


class WorkerState(StrEnum):
    READY = "ready"
    BUSY = "busy"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class WorkerInfo:
    """A worker's advertised identity and available capacity."""

    worker_id: str
    address: str
    memory_bytes: int = 0
    state: WorkerState = WorkerState.READY
    labels: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class WeightShard:
    """Metadata describing one model-weight shard assigned to a worker."""

    model_id: str
    tensor_name: str
    shard_index: int
    shard_count: int
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    """Input to a generation run; tokenization and scheduling are added later."""

    request_id: str
    prompt: str
    max_new_tokens: int = 128


@dataclass(frozen=True, slots=True)
class TokenEvent:
    """One streamed output event sent from the engine to a client."""

    request_id: str
    token: str
    finished: bool = False
