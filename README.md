# Distributed Memory Grid for Real-Time LLM Inference

An early-stage project exploring how to coordinate model shards across commodity worker nodes. The current review milestone focuses on deterministic tensor layout helpers, a lightweight coordinator API, and framework-independent runtime sketches.

## Current status

**Implemented:** column- and row-parallel tensor split/compute/gather demonstrations; shard metadata validation; an in-memory worker registry; FastAPI health, registration, worker listing, and cluster summary routes; small continuous-batch admission and bounded token-slot cache primitives.

**Not implemented yet:** gRPC transport, PyTorch distributed collectives, safetensors loading, real model execution, GPU KV-cache/page management, token streaming, multi-node deployment, or performance measurements. The tensor demos use small synthetic values and run in one process. The website's flow animation is also a local simulation, not a live cluster view.

This is a review prototype, not a production inference service. See [the architecture](docs/ARCHITECTURE.md), [the four-week roadmap](docs/ROADMAP.md), and [the review website](website/README.md).

## Setup and run

Requires Python 3.11 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
python -m pip install -e '.[dev]'
PYTHONPATH=src python -m distributed_memgrid.demo
PYTHONPATH=src python -m distributed_memgrid.row_parallel_demo
```

Start the coordinator API with `PYTHONPATH=src python -m distributed_memgrid.master`. It listens on `http://127.0.0.1:8000` and exposes:

- `GET /health` — process health.
- `POST /workers` — register a worker with `worker_id`, `address`, optional `memory_bytes`, and optional `labels`.
- `POST /workers/{worker_id}/heartbeat` — report liveness and current `state` (`ready`, `busy`, or `unavailable`). Workers that do not heartbeat for 30 seconds are reported as unavailable.
- `GET /workers` — list registered workers.
- `GET /cluster` — summarize worker count and reported memory.

Example registration body:

```json
{"worker_id":"worker-01","address":"127.0.0.1:50051","memory_bytes":8589934592,"labels":{"zone":"local"}}
```

The registry is in memory and resets when the process restarts. Registration is a coordinator API boundary; it is not a gRPC worker connection.

## Review website

Preview the static site with `python3 -m http.server 8080 --directory website` and open `http://localhost:8080`. The site is deployable through the GitHub Pages workflow in `.github/workflows/pages.yml`; see [website deployment notes](website/README.md). Hosting requires enabling GitHub Pages on the repository, and no public URL exists until that workflow is run successfully.

## Team workflow

- Keep `main` stable and work on one branch per contributor.
- Push meaningful work regularly; do not use empty commits to satisfy activity requirements.
- Open a pull request for changes to `main` and request peer review before merging.
- Track tasks in GitHub Issues and keep `docs/WORK_DISTRIBUTION.md` current.
- Never commit model weights or credentials.
