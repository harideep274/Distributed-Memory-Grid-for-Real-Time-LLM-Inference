# Distributed Memory Grid for Real-Time LLM Inference

Team project to explore sharded model weights and distributed inference across worker nodes.

## Project status

**Week 1: architecture and foundation.** The repository currently contains the agreed interfaces and team workflow. The first end-to-end milestone is a master that can register workers and report their status; model loading and distributed execution follow after that boundary is reviewed by the team.

## Team workflow

- Keep `main` stable; work on one branch per member (`lead`, `networking`, `model-parallel`, `runtime-api`). Replace these example names with your GitHub usernames.
- Push meaningful work daily. Do not use empty commits to satisfy activity requirements.
- Open a pull request for changes to `main`; request a peer review before merging.
- Track work in GitHub Issues and keep `docs/WORK_DISTRIBUTION.md` current.

## Setup

Requires Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
python -m pip install -e '.[dev]'
```

Run the initial master process:

```bash
python -m distributed_memgrid.master
```

The master currently exposes a small HTTP health and worker-status endpoint. This is a development scaffold, not yet a distributed inference server. The team will replace/extend the transport with the gRPC contract in the Week 1 implementation.

## Project plan

See [the architecture](docs/ARCHITECTURE.md), [four-person work distribution](docs/WORK_DISTRIBUTION.md), and [weekly milestones](docs/ROADMAP.md).
