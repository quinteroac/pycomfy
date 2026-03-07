# Agents entry point

- **What this project is:** (Describe your product or project. One or two sentences. This file is the single entry point for the agent.)
- **How to work here:** Use this file as the single entry point. Follow the process phases in order; read and update `.agents/state.json` for the current iteration and phase. Invoke the skills under `.agents/skills/` as indicated by each NVST command. All iteration artifacts live in `.agents/flow/` with the naming `it_` + 6-digit iteration (e.g. `it_000001_product-requirement-document.md`). From the second iteration onward, adhere to [`.agents/PROJECT_CONTEXT.md`](.agents/PROJECT_CONTEXT.md).
- **Process:** Define → Prototype → Refactor (see `docs/nvst-flow/` or package documentation).
- **Project context:** [`.agents/PROJECT_CONTEXT.md`](.agents/PROJECT_CONTEXT.md) — conventions and architecture; the agent adheres from the second iteration onward.
- **Rule:** All generated resources in this repo must be in English.
