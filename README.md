# pycomfy

A standalone Python library that exposes ComfyUI’s inference engine as importable modules—no server, no unnecessary nodes, no application layer. Designed to be consumed like [diffusers](https://github.com/huggingface/diffusers) or [diffsynth](https://github.com/Comfy-Org/diffsynth): import and run inference in your own code.

## Goals

- **Library-first:** Use ComfyUI’s execution and model loading as a Python API, not as a web UI or long-lived server.
- **Minimal surface:** Only the inference core (graph execution, model loading, device handling). No default node set, no bundled workflows.
- **Familiar usage:** Same mental model as `diffusers` / `diffsynth`: load a pipeline or model, call a method, get tensors or images back.

## Status

Early stage. Structure and entry points are being defined.

## Usage (target)

```python
from pycomfy import load_pipeline, run

pipeline = load_pipeline("path/to/workflow.json", base_path="path/to/models")
outputs = run(pipeline, inputs={"prompt": "a cat"})
# outputs: dict of tensors/images per output node
```

## Tooling

- **Python:** Managed with [uv](https://docs.astral.sh/uv/). Install dependencies and run the project with uv; do not use pip/venv for this repo.
- **NVST (agent workflow):** The Define → Prototype → Refactor process and NVST commands are run with [Bun](https://bun.sh). Use `bun nvst <command>` (see `docs/nvst-flow/`).

### Setup

```bash
# Python (uv)
uv sync

# NVST / agent toolkit (bun)
bun install
bun nvst init          # first time only
bun nvst start iteration
```

### Common commands

```bash
uv run python -m pycomfy ...   # run the library
uv add <package>               # add Python dependency
bun nvst flow --agent cursor   # run next NVST step
```

## Development

See [AGENTS.md](AGENTS.md) for the agent workflow and project context.
