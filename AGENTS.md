# Agents entry point

- **What this project is:** A standalone Python library that exposes ComfyUI’s inference engine as importable modules—no server, no unnecessary nodes, no application layer. It is meant to be consumed like diffusers or diffsynth: import and run inference in your own code.
- **How to work here:** Use this file as the single entry point. Follow the process phases in order; read and update `.agents/state.json` for the current iteration and phase. Invoke the skills under `.agents/skills/` as indicated by each NVST command. All iteration artifacts live in `.agents/flow/` with the naming `it_` + 6-digit iteration (e.g. `it_000001_product-requirement-document.md`). From the second iteration onward, adhere to [`.agents/PROJECT_CONTEXT.md`](.agents/PROJECT_CONTEXT.md). **Python:** use [uv](https://docs.astral.sh/uv/) for install, run, and adding dependencies (`uv sync`, `uv run`, `uv add`). **NVST:** run all agent/workflow commands with [Bun](https://bun.sh) as `bun nvst <command>` (see `docs/nvst-flow/`).
- **Process:** Define → Prototype → Refactor (see `docs/nvst-flow/` or package documentation).
- **Project context:** [`.agents/PROJECT_CONTEXT.md`](.agents/PROJECT_CONTEXT.md) — conventions and architecture; the agent adheres from the second iteration onward.
- **Rule:** All generated resources in this repo must be in English.
