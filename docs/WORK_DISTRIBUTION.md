# Two-week work distribution — Project 1

Replace role labels with the four members' names and GitHub usernames at kickoff. Keep `main` stable and use the listed branches (or replace them with the members' GitHub usernames).

## Shared two-week goal

By the end of Week 2, demonstrate a master registering and monitoring multiple workers, assigning model tensor shards, and having workers load and validate those shards. Include a reproducible small-model or synthetic-tensor demo, correctness evidence against the unsharded tensors, and setup/run instructions. Do not commit model weights. Treat a real multi-node generation service as a later milestone unless the team confirms the hardware and model support early.

| Member | Branch | Ownership | Week 1 deliverables | Week 2 deliverables |
|---|---|---|---|---|
| You | `model-parallel` | Core technical owner: weight loading, tensor-parallel layouts, shard correctness | Specify supported tensor layouts and divisibility/padding rules; implement safetensors or fixture loading, metadata checks, and row/column shard split/join primitives | Integrate assigned shards with workers; verify reconstructed tensors and partitioned linear results against a single-process reference; document memory behavior and unsupported cases |
| Member 2 | `networking` | Master/worker lifecycle and transport | Implement worker registration, heartbeat/health and status reporting; document RPC contract | Complete master-to-worker shard assignment path and reconnect/failure behavior; demonstrate multiple local workers |
| Member 3 | `runtime-api` | Request boundary and runnable demo | Define minimal request/result types and coordinator interface; add demo entry point | Connect coordinator to registered workers and assigned shards; expose a small status/demo API with clear unsupported-feature responses |
| Member 4 | `integration` | Packaging, integration support, demo reliability | Establish local multi-worker launch configuration and setup guide; review shared message types with owners | Integrate PRs, add failure/invalid-input handling at boundaries, and prepare clean-start demo and results notes |

## Two-week cadence

- **Day 1:** Agree on the demo model/fixture, hardware, acceptance checks, member names and branch names. You publish the tensor-layout contract; create one issue per deliverable and freeze shared interfaces.
- **Days 2–4:** Implement each owner's Week 1 work. You build and document the shard primitives while the networking and runtime owners build to the agreed interfaces. Push working increments daily and review interfaces before building on them.
- **Day 5:** Integrate registration and shard metadata; run a short design review and adjust Week 2 scope based on blockers.
- **Days 6–9:** You complete tensor layout integration and compare distributed shard results with a single-process reference. Networking completes assignment delivery; runtime and integration owners wire and launch the demo. Pair across ownership boundaries when integration fails.
- **Day 10:** Freeze features, run the demo from a clean setup, capture correctness evidence, and present known limits and next steps.

## Coordination rules

- All four members contribute throughout; ownership identifies the first responsible person, not an exclusive boundary.
- Keep changes on individual branches, push meaningful work daily, and open reviewed pull requests into `main`.
- Keep interfaces small and agree them before parallel implementation. The lead resolves integration conflicts with the relevant owner.
- Update issue status and record decisions, review evidence, blockers, and revised assignments here or in linked GitHub Issues.
- SOP response expectation: within 24 hours. At 48 hours without response, follow the SOP contact/reporting steps and reassign work so the project continues.
