# Four-week roadmap

## Week 1 — topology and weight loading

- Finalize worker registration, health checking, and shard-assignment contracts.
- Implement the asyncio/gRPC master-worker path.
- Load safetensors weights and inspect memory usage; keep weights outside Git.
- Midweek: review interfaces. End of week: demonstrate master and worker communication.

## Week 2 — tensor parallelism and KV cache

- Implement and review tensor splitting for attention and feed-forward layers.
- Verify shard shapes and reconstruct/compare against the original tensors to detect data loss.
- Add asynchronous KV-cache allocation and release behavior.
- Mid-project review: demonstrate topology and correct weight distribution.

## Week 3 — generation serving

- Add iteration-level continuous batching.
- Add the FastAPI/WebSocket token-streaming path.
- Measure queueing and inter-node latency; record limitations.

## Week 4 — deployment and evaluation

- Containerize the demo and document startup of master and workers.
- Run repeatable load measurements for time to first token, throughput, and inter-node latency.
- Document hardware, model, concurrency, configuration, results, and known limits.
- Final review: demonstrate distributed generation and present the report/deployment steps.

## GitHub cadence from the SOP

- Daily meaningful pushes across the project.
- Mid review: repository commits on at least 10 distinct days in the preceding 14 days.
- Final review: a repository commit on each day in the preceding 20-day period.
- Confirm the exact review dates with the program mentor and keep the team present for reviews.
