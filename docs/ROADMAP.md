# Four-week project roadmap

This roadmap translates the project brief into staged acceptance evidence. The repository currently contains an early review prototype; stages marked planned require implementation and measurement before the final project claims can be made.

## Week 1 — topology and weight path

- Freeze worker registration, heartbeat, and shard-assignment RPC contracts.
- Replace coordinator-only HTTP registration with gRPC worker registration and health checks.
- Add safetensors metadata inspection and memory-mapped shard loading; keep model files external to Git.
- Demonstrate two worker processes and capture registration and shard assignment logs.

**Checkpoint:** two workers register, report capacity, receive valid assignments, and reject malformed metadata.

## Week 2 — tensor parallelism and cache lifecycle

- Integrate PyTorch tensor layouts with attention and feed-forward layer partitions.
- Compare partitioned operations to a single-process reference for selected shapes and dtypes.
- Replace the token-slot runtime sketch with a model-backed KV-cache allocator and defined reclamation behavior.
- Require peer review for synchronization and cache changes.

**Checkpoint:** supported operations reconstruct or match the baseline within documented tolerance; cache memory is accounted for under concurrent sequences.

## Week 3 — request scheduling and streaming

- Implement iteration-level continuous batching with cancellation and per-request limits.
- Connect a supported model path to worker execution and collective communication; use NCCL only where the hardware supports it.
- Expose a FastAPI/WebSocket client interface and stream generated token events.
- Measure basic throughput and time to first token on a documented profile.

**Checkpoint:** requests can enter and leave active batches while other sequences continue, and a client can consume a real token stream.

## Week 4 — deployment and evaluation

- Package coordinator and workers in containers and deploy to a documented local or cloud cluster.
- Configure node placement and health recovery for the selected orchestration option.
- Run repeatable concurrency and failure scenarios; report TTFT, throughput, and inter-node latency.
- Finalize operating instructions, architecture report, known limitations, and project review materials.

**Checkpoint:** a reproducible deployment generates tokens across workers, with benchmark methodology and results published alongside deployment scripts.

## Progress update — synthetic worker assignment slice

The repository now has a runnable `distributed_memgrid.cluster_demo` that
registers two in-process workers, assigns each a synthetic column-parallel
weight shard, validates and retains the shard, computes local outputs, and
checks that gathering matches the unsharded reference. Worker-side validation
rejects malformed rank, shape, axis, and shard index metadata.

This is partial evidence for the shared two-week goal. It does not complete
Week 1: workers are still in one process, assignments use Python objects, and
there is no gRPC transport, checkpoint loading, reconnect behavior, or
multi-process run configuration. The next concrete step is to freeze an RPC
schema and move registration and shard assignment across a local process
boundary before adding safetensors loading.

## Current review milestone

The current code provides dependency-free synthetic column- and row-parallel linear demos, coordinator HTTP endpoints, and in-memory runtime sketches. It does not satisfy the multi-node inference checkpoints above. Use the four-week review website as a status overview, and update this file as each checkpoint gains evidence.
