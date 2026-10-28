# Initial architecture

## Goal

Demonstrate a master coordinating model shards across worker processes, with a client API for streaming generated tokens. The first milestone is a small, observable master/worker topology. Full model execution is added incrementally and must be verified against a single-node baseline.

## Components

- **Master / coordinator:** maintains worker membership, places model shards, and later schedules generation work.
- **Workers:** load assigned safetensors shards with memory mapping where supported and participate in tensor-parallel operations.
- **Inference runtime:** owns request scheduling, continuous batching, and per-request KV-cache state.
- **API gateway:** accepts generation requests and streams token events to clients using FastAPI/WebSockets.
- **Transport:** gRPC for master/worker RPCs; the API gateway uses HTTP and WebSockets. Transport message definitions live in `proto/` once the team agrees on the first RPC contract.

## Initial design decisions

1. Keep message types transport-independent in `protocol.py` so RPC serialization can be changed without rewriting the runtime.
2. Start with explicit worker registration and status reporting; add health checks and failure handling before relying on a worker for inference.
3. Keep model execution optional until the topology and shard assignment are demonstrable. Do not check model weights into Git.
4. Start with a small model and a documented hardware profile. Record any deviations from the project brief with the mentor.
5. Treat generation output as a stream of `TokenEvent` values; the API must not depend on model-specific internals.

## Week 1 acceptance evidence

- A master starts and exposes health/status information.
- Workers can be identified and their capacity reported.
- The team agrees on and documents the gRPC registration and shard-assignment RPCs.
- A supported small model can be loaded from safetensors without adding model files to the repository.
