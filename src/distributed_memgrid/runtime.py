"""Small, framework-independent primitives for request and KV-cache scheduling.

These primitives make the runtime boundary reviewable. They are not a token
generation engine and do not claim paged attention or GPU memory management.
"""

from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass, field

from distributed_memgrid.protocol import GenerationRequest


@dataclass(slots=True)
class RequestState:
    request: GenerationRequest
    generated_tokens: int = 0
    finished: bool = False


class ContinuousBatchScheduler:
    """Iteration-level FIFO admission with a configurable active-request cap."""

    def __init__(self, max_batch_size: int = 8) -> None:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be positive")
        self.max_batch_size = max_batch_size
        self._pending: deque[RequestState] = deque()
        self._active: OrderedDict[str, RequestState] = OrderedDict()

    def submit(self, request: GenerationRequest) -> None:
        if not request.request_id.strip():
            raise ValueError("request_id cannot be empty")
        if request.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if request.request_id in self._active or any(
            item.request.request_id == request.request_id for item in self._pending
        ):
            raise ValueError(f"duplicate request_id: {request.request_id}")
        self._pending.append(RequestState(request))

    def next_batch(self) -> list[RequestState]:
        """Admit pending work and return the active set for one decode step."""
        while self._pending and len(self._active) < self.max_batch_size:
            item = self._pending.popleft()
            self._active[item.request.request_id] = item
        return list(self._active.values())

    def complete_step(self, request_id: str) -> RequestState:
        """Record one generated token and retire a request at its token limit."""
        try:
            state = self._active[request_id]
        except KeyError as exc:
            raise KeyError(f"request is not active: {request_id}") from exc
        state.generated_tokens += 1
        if state.generated_tokens >= state.request.max_new_tokens:
            state.finished = True
            del self._active[request_id]
        return state

    @property
    def pending_count(self) -> int:
        return len(self._pending)


@dataclass(slots=True)
class KVCache:
    """Bounded token-slot cache model for lifecycle and eviction demonstrations."""

    capacity_tokens: int
    _entries: OrderedDict[str, list[int]] = field(default_factory=OrderedDict)
    _used_tokens: int = 0

    def __post_init__(self) -> None:
        if self.capacity_tokens < 1:
            raise ValueError("capacity_tokens must be positive")

    @property
    def used_tokens(self) -> int:
        return self._used_tokens

    def append(self, request_id: str, token_id: int) -> None:
        if request_id not in self._entries:
            self._entries[request_id] = []
        if self._used_tokens == self.capacity_tokens:
            raise MemoryError("KV cache capacity reached; release a request before appending")
        self._entries[request_id].append(token_id)
        self._used_tokens += 1
        self._entries.move_to_end(request_id)

    def release(self, request_id: str) -> int:
        tokens = self._entries.pop(request_id, None)
        if tokens is None:
            return 0
        self._used_tokens -= len(tokens)
        return len(tokens)

    def snapshot(self) -> dict[str, object]:
        return {
            "capacity_tokens": self.capacity_tokens,
            "used_tokens": self._used_tokens,
            "requests": {key: list(value) for key, value in self._entries.items()},
        }
