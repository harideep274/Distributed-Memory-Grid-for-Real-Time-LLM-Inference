# Two-week roadmap

## Target demo

Show a master coordinating at least two worker processes on one development machine. Workers register and report status; the coordinator assigns each a portion of model tensors; workers load the assigned shards; the team verifies that the shards reconstruct the source tensors. Use a small supported safetensors model when hardware permits, otherwise use deterministic synthetic tensors and document the substitution. Model weights stay outside Git.

This sprint proves the control plane and shard correctness. It does not promise production-ready distributed token generation, continuous batching, or multi-node performance.

## Week 1 — interfaces and worker topology

- **Day 1:** Confirm team names, branches, development environment, model/fixture, demo hardware, and acceptance checks. Freeze registration and shard-assignment message shapes.
- **Days 2–3:** Implement worker registration, status, heartbeat/health, and local multi-worker startup.
- **Days 2–4:** You implement tensor metadata validation, safetensors or synthetic tensor loading, and row/column split/join helpers with documented layout rules. In parallel, the team implements worker registration and the demo boundary.
- **Day 4:** Define the minimal coordinator/request boundary and the runnable demo entry point.
- **Day 5 checkpoint:** Master sees two workers, reports their status, and the team can serialize/validate a shard assignment. Review interfaces and risks before Week 2.

## Week 2 — shard path and integrated demonstration

- **Days 6–7:** Implement assignment delivery and worker-side shard loading; reject invalid model IDs, shapes, dtypes, and shard indices clearly. You integrate layouts with loaded tensors and document unsupported cases.
- **Days 7–8:** Integrate the coordinator with worker registry and shard loader. Add a small status/demo API if it helps show the result.
- **Day 9:** Compare shard reconstruction and partitioned linear results against a single-process reference on representative tensor shapes and dtypes; run the full demo from a clean setup. Fix integration defects; avoid adding new scope.
- **Day 10:** Present the demo, setup/run instructions, correctness evidence, memory observations, known limitations, and a prioritized next milestone.

## Acceptance checks

- The master starts and exposes health and worker status.
- At least two worker processes register and can be observed by the master.
- A shard assignment reaches the intended worker and invalid assignments are rejected.
- Supported source tensors can be split and reconstructed with documented tolerance, shapes, and dtypes; partitioned linear results match a single-process reference within tolerance.
- A fresh checkout can follow the documented setup and run the demo without model weights stored in the repository.
- The demo report records environment, model or fixture, steps run, results, and limitations.

## Follow-on work after the two-week sprint

Add a real model execution path, tensor-parallel operator integration, request scheduling and KV-cache management, token streaming, multi-node deployment, and repeatable latency/throughput measurements. Sequence these after the shard-path demo is stable and hardware/model constraints are known.
