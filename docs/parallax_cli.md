# @parallax/cli

Bun-based command-line interface for the Parallax platform. Provides commands to generate images/video, edit media, and manage models — all by talking to `@parallax/ms`.

## Role in the monorepo

End-user interface for local usage. Does not call `comfy_diffusion` directly — it sends HTTP requests to the Elysia gateway and streams SSE events for job progress.

## Location

`packages/parallax_cli/`

## Stack

- Bun runtime
- `commander` for command parsing
- `@parallax/sdk` for shared request/response types

## Running

```bash
# from repo root
bun run cli

# direct
cd packages/parallax_cli && bun run src/index.ts
```

## Planned Commands

```
parallax generate   --prompt <text> [--model <id>] [--steps <n>] [--output <path>]
parallax edit       --image <path> --prompt <text> [--steps <n>]
parallax models     list
parallax jobs       status <id>
```

## Dependencies

```json
{
  "commander": "^12.0.0",
  "@parallax/sdk": "workspace:*"
}
```
