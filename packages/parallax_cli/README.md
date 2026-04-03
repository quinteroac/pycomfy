# @parallax/cli

Command-line interface for the Parallax image-generation pipeline.

## Installation

```bash
# From the monorepo root
bun install
```

## Usage

```bash
# Run directly with bun
bun run packages/parallax_cli/src/index.ts --help

# Or via the bin entry-point after linking (see Global installation below)
parallax --help
```

### Commands

| Command | Description |
|---------|-------------|
| `generate` | Generate an image from a text prompt |
| `edit` | Edit an existing image with a prompt |

## Global installation (US-008)

To invoke `parallax` as a global command you need to link the package with `bun link`.

### Steps

1. **Link the package** (run once from the `parallax_cli` directory):

   ```bash
   cd packages/parallax_cli
   bun link
   ```

2. **Verify the link**:

   ```bash
   which parallax    # should resolve to the bun shims directory
   parallax --help   # should print the CLI help text
   ```

3. **Smoke test** (AC01 / AC02 verification):

   ```bash
   # AC01 – global command resolves
   parallax --version

   # AC02 – a real generate call succeeds (requires a running Parallax ms backend)
   parallax generate "a red cube on a white background"
   ```

4. **Unlink** when finished:

   ```bash
   bun unlink @parallax/cli
   ```

> **Note:** `bun link` / global invocation is a manual verification step and is not covered by the automated test suite.

## Development

```bash
# Watch mode
bun run dev

# Type-check only
bun run typecheck
```
