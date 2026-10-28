# Distributed Memory Grid for Real-Time LLM Inference

Team project to explore sharded model weights and distributed inference across worker nodes.

## Project status

**Two-week restart sprint: control plane and shard correctness.** The repository is a scaffold. The target demo is a master registering multiple workers, assigning model tensor shards, and validating shard loading and reconstruction. Full distributed generation is a follow-on milestone. See the two-week roadmap and four-person work distribution below.

## Team workflow

- Keep `main` stable; work on one branch per member (`lead`, `networking`, `model-parallel`, `runtime-api`). Replace these example names with your GitHub usernames.
- Push meaningful work daily. Do not use empty commits to satisfy activity requirements.
- Open a pull request for changes to `main`; request a peer review before merging.
- Track work in GitHub Issues and keep `docs/WORK_DISTRIBUTION.md` current.

## Setup

Requires Python 3.11 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
python -m pip install -e '.[dev]'
```

Run the initial master process:

```bash
python -m distributed_memgrid.master
```

The master currently exposes a small HTTP health and worker-status endpoint. This is a development scaffold, not yet a distributed inference server. Agree on the first master/worker RPC contract before implementing the transport.

## Run the model-parallel demo

The sharding demo uses synthetic tensors, so it needs no model download or optional ML dependencies. From the repository root, install the package in editable mode and run:

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e .
python -m distributed_memgrid.demo
```

To run it without installing the package, use `PYTHONPATH=src python3 -m distributed_memgrid.demo` from the repository root. In VS Code, open this repository folder, select your Python interpreter, then choose **Run and Debug → Run model-parallel demo** (F5).

It partitions a linear layer's weight matrix by output columns, simulates each worker's local matrix multiply, gathers the outputs, and compares them with a single-process reference. The initial implementation is pure Python and demonstrates the layout and correctness boundary; it does not yet load a Hugging Face model or communicate with separate worker processes.

## Project plan

See [the architecture](docs/ARCHITECTURE.md), [four-person work distribution](docs/WORK_DISTRIBUTION.md), and [two-week roadmap](docs/ROADMAP.md).
