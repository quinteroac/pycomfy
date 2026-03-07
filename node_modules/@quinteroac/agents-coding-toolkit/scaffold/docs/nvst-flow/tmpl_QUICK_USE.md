# NVST Quick Use

Quick reference for common workflows. See [COMMANDS.md](COMMANDS.md) for full command reference.

**Running commands:** Use `bun nvst` so Bun resolves the binary. Or add `node_modules/.bin` to your PATH: `export PATH="$PATH:$(pwd)/node_modules/.bin"` to run `nvst` directly.

## First-time setup

```bash
# Install the toolkit from npm (package is published)
npm install @quinteroac/agents-coding-toolkit
# or with Bun
bun add @quinteroac/agents-coding-toolkit

# From local path (development)
# bun add /path/to/nerds-vibecoding-survivor-toolkit

# Initialize scaffold in your project
bun nvst init

# Start the first iteration
bun nvst start iteration
```

## Typical iteration flow

### 1. Define

```bash
bun nvst define requirement --agent codex
# Review, then optionally:
bun nvst refine requirement --agent codex --challenge
bun nvst approve requirement
# PRD JSON is created when needed (e.g. via create prototype)
```

### 2. Prototype

```bash
bun nvst create project-context --agent codex --mode yolo
bun nvst approve project-context

bun nvst create prototype --agent codex --iterations 10

bun nvst define test-plan --agent codex
bun nvst refine test-plan --agent codex   # optional
bun nvst approve test-plan

bun nvst execute test-plan --agent codex

# If tests fail:
bun nvst execute automated-fix --agent codex
# or for manual debugging:
bun nvst execute manual-fix --agent codex

# When all pass:
bun nvst approve prototype
```

### 3. Refactor

```bash
bun nvst define refactor-plan --agent codex
bun nvst refine refactor-plan --agent codex   # optional
bun nvst approve refactor-plan

bun nvst execute refactor --agent codex

# Run all tests, update PROJECT_CONTEXT.md, record CHANGELOG.md
# Then start next iteration:
bun nvst start iteration
```

## Flow command (semi-automated)

`bun nvst flow` runs the next pending step(s) until it hits an approval gate:

```bash
bun nvst flow --agent codex
```

Use `--force` to bypass guardrail confirmations.

## Agent providers

Use `--agent` with: `claude`, `codex`, `gemini`, or `cursor`.

Example:

```bash
bun nvst define requirement --agent cursor
```
