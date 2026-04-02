# @parallax/sdk

Shared TypeScript types and API contracts used across `@parallax/ms`, `@parallax/cli`, and `@parallax/mcp`. Acts as the single source of truth for request/response shapes.

## Role in the monorepo

Prevents type drift between packages. Because all three TypeScript packages reference `@parallax/sdk` via `workspace:*`, changing a type here causes compile errors in all consumers — making breaking changes visible immediately.

## Location

`packages/parallax_sdk/`

## Stack

- TypeScript ^5.0
- No runtime dependencies — types only

## Usage

```typescript
import type { GenerateImageRequest, GenerateImageResponse, EditImageRequest } from "@parallax/sdk";
```

## Current Types

```typescript
// GenerateImageRequest
{
  prompt: string;
  negative_prompt?: string;
  width?: number;
  height?: number;
  steps?: number;
}

// EditImageRequest
{
  image_path: string;
  prompt: string;
  steps?: number;
}

// GenerateImageResponse
{
  image_path: string;
  seed: number;
}
```

## Extending

Add new types to `src/types.ts` and re-export from `src/index.ts`. Keep types serialization-friendly (no `Date`, `Map`, `Set` — use primitives and plain objects).

## Dependencies

None. This package has no runtime dependencies by design.
