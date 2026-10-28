# Work distribution — Project 1

Replace role labels with the four members' names and GitHub usernames before submitting this document.

| Member | Branch | Ownership | Week 1 | Week 2 | Week 3 | Week 4 |
|---|---|---|---|---|---|---|
| Team lead (you) | `lead` | Architecture, repository, integration, reviews | Repository and interfaces; coordinate topology | Integrate parallelism; conduct peer reviews | Integrate API and scheduler | Deployment, measurements, final report/review |
| Member 2 | `networking` | asyncio/gRPC and worker lifecycle | Master/worker communication | RPC support for shard placement | Streaming path integration | Deployment and latency measurements |
| Member 3 | `model-parallel` | Safetensors and tensor parallelism | Weight loading and memory mapping | Shard attention/FFN tensors; correctness evidence | Runtime integration | End-to-end correctness and resource measurements |
| Member 4 | `runtime-api` | KV cache, batching, API | Define request interfaces | KV-cache lifecycle | Continuous batching and FastAPI/WebSocket gateway | Load runs and operational docs |

## Coordination rules

- All four members contribute throughout the project; ownership identifies the first responsible person, not an exclusive boundary.
- Each person works on their own branch and pushes meaningful changes daily.
- Use pull requests and peer review for integration into `main`.
- Keep task updates, decisions, review evidence, and any revised assignments in this document or linked GitHub Issues.
- SOP response expectation: within 24 hours. At 48 hours without response, follow the SOP contact/reporting steps and reassign work so the project continues.
