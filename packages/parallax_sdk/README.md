# @parallax/sdk

Shared TypeScript SDK used by `@parallax/cli`, `@parallax/ms`, and `@parallax/mcp`. Provides job queue access, submission, status polling, listing, cancellation, health checks, and shared types.

The queue is backed by [bunqueue](https://github.com/nicolo-ribaudo/bunqueue) with an embedded SQLite database at `~/.config/parallax/jobs.db`.

## Installation

This package is consumed from within the monorepo as a workspace dependency:

```json
"@parallax/sdk": "workspace:*"
```

## Modules

### `@parallax/sdk` (barrel)

Re-exports everything from all modules below.

---

### `@parallax/sdk/types`

Shared request/response types for image, video, and audio operations.

```ts
import type {
  GenerateImageRequest,
  GenerateImageResponse,
  EditImageRequest,
  GenerateVideoRequest,
  GenerateVideoResponse,
  GenerateAudioRequest,
  GenerateAudioResponse,
} from "@parallax/sdk/types";
```

---

### `@parallax/sdk/jobs`

Job data and result types shared across all packages.

```ts
import type { ParallaxJobData, ParallaxJobResult, PythonProgress } from "@parallax/sdk/jobs";
```

| Type | Fields |
|------|--------|
| `ParallaxJobData` | `action`, `media`, `model`, `script`, `args`, `scriptBase`, `uvPath` |
| `ParallaxJobResult` | `outputPath` |
| `PythonProgress` | `step`, `pct`, `frame?`, `total?`, `output?`, `error?` |

---

### `@parallax/sdk/queue`

Singleton queue instance (bunqueue embedded SQLite).

```ts
import { getQueue } from "@parallax/sdk/queue";

const queue = getQueue();  // ~/.config/parallax/jobs.db
await queue.close();
```

---

### `@parallax/sdk/submit`

Submit a new inference job to the queue. Spawns a detached worker process to execute the job.

```ts
import { submitJob } from "@parallax/sdk/submit";

const jobId = await submitJob({
  action: "create",
  media: "image",
  model: "sdxl",
  script: "runtime/image/generation/sdxl/t2i.py",
  args: ["--prompt", "a red cube", "--output", "output.png"],
  scriptBase: "/path/to/repo",
  uvPath: "uv",
});
// → "42"
```

---

### `@parallax/sdk/status`

Get the current status of a job by ID.

```ts
import { getJobStatus } from "@parallax/sdk/status";
import type { ParallaxJobStatus } from "@parallax/sdk/status";

const status = await getJobStatus("42");
// → {
//     id: "42",
//     status: "completed",   // "waiting" | "active" | "completed" | "failed"
//     progress: 100,
//     model: "sdxl",
//     action: "create",
//     media: "image",
//     output: "output.png",
//     error: null,
//     createdAt: 1712000000000,
//     startedAt: 1712000001000,
//     finishedAt: 1712000019000,
//   }
// → null if not found
```

---

### `@parallax/sdk/list`

List jobs from the queue with optional status filter and limit.

```ts
import { listJobs } from "@parallax/sdk/list";
import type { JobSummary, JobListResult, JobStatusValue } from "@parallax/sdk/list";

const result = await listJobs({ status: "completed", limit: 20 });
// result.jobs  → JobSummary[]  (sorted newest first)
// result.counts → { waiting, active, completed, failed }
```

`JobSummary` fields: `id`, `status`, `progress`, `model`, `action`, `media`, `createdAt`, `duration` (ms or `null`).

---

### `@parallax/sdk/cancel`

Cancel a waiting or active job.

```ts
import { cancelJob } from "@parallax/sdk/cancel";
import type { CancelJobOutcome } from "@parallax/sdk/cancel";

const outcome = await cancelJob("42");
// true          → cancelled successfully
// null          → job not found
// "completed"   → already completed (no-op)
// "failed"      → already failed (no-op)
```

---

### `@parallax/sdk/health`

Get aggregate queue statistics.

```ts
import { getQueueStats } from "@parallax/sdk/health";
import type { QueueStats } from "@parallax/sdk/health";

const stats = await getQueueStats();
// → { waiting: 0, active: 1, completed: 42, failed: 2 }
```

---

## Development

```bash
bun run typecheck   # Type-check only
```
