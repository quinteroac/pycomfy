import { Elysia, t } from "elysia";
import { cors } from "@elysiajs/cors";
import { Stream } from "@elysiajs/stream";
import { join } from "path";
import { submitJob, getJobStatus, listJobs, cancelJob, getQueueStats } from "@parallax/sdk";
import type { JobStatusValue } from "@parallax/sdk";
import type { ParallaxJobData } from "@parallax/sdk";
import {
  getScript,
  getModelConfig,
} from "../../parallax_cli/src/models/registry";

function getScriptBase(): string {
  return process.env.PARALLAX_RUNTIME_DIR ?? process.env.PARALLAX_REPO_ROOT ?? process.cwd();
}

function getUvPath(): string {
  return process.env.PARALLAX_UV_PATH ?? "uv";
}

export const app = new Elysia()
  .use(cors())
  .get("/health", async () => {
    const queue = await getQueueStats();
    return { status: "ok", queue };
  })
  .onError(({ code, error, set }) => {
    if (code === "VALIDATION") {
      set.status = 400;
      return { error: error.message };
    }
    set.status = 500;
    return { error: "Internal server error" };
  })

  // US-004: GET /jobs — list recent jobs with counts
  .get(
    "/jobs",
    async ({ query }) => {
      const status = query.status as JobStatusValue | undefined;
      return listJobs({ status });
    },
    {
      query: t.Object({
        status: t.Optional(
          t.Union([
            t.Literal("waiting"),
            t.Literal("active"),
            t.Literal("completed"),
            t.Literal("failed"),
          ]),
        ),
      }),
    },
  )

  // US-003: GET /jobs/:id/stream — Server-Sent Events until job completion
  .get("/jobs/:id/stream", async ({ params, set }) => {
    const initial = await getJobStatus(params.id);
    if (!initial) {
      set.status = 404;
      return { error: "Job not found" };
    }

    const id = params.id;
    const stream = new Stream<object>(async (s) => {
      while (true) {
        await s.wait(500);
        const status = await getJobStatus(id);

        if (!status) {
          s.close();
          break;
        }

        if (status.status === "completed") {
          s.event = "completed";
          s.send({ output: status.output });
          s.close();
          break;
        } else if (status.status === "failed") {
          s.event = "failed";
          s.send({ error: status.error });
          s.close();
          break;
        } else {
          s.event = "progress";
          s.send({ pct: status.progress, step: status.status });
        }
      }
    });

    return new Response(stream.value, {
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
      },
    });
  })

  // US-002: GET /jobs/:id — poll job status
  .get("/jobs/:id", async ({ params, set }) => {
    const status = await getJobStatus(params.id);
    if (!status) {
      set.status = 404;
      return { error: "Job not found" };
    }
    return status;
  })

  // US-005: DELETE /jobs/:id — cancel a running or waiting job
  .delete("/jobs/:id", async ({ params, set }) => {
    const result = await cancelJob(params.id);
    if (result === null) {
      set.status = 404;
      return { error: "Job not found" };
    }
    if (result !== true) {
      set.status = 409;
      return { error: "Job already completed" };
    }
    return { cancelled: true };
  })

  // AC01: POST /jobs/create/image
  .post(
    "/jobs/create/image",
    async ({ body }) => {
      const args: string[] = ["--prompt", body.prompt, "--output", "output.png"];
      if (body.negative_prompt) args.push("--negative-prompt", body.negative_prompt);
      if (body.width != null)   args.push("--width",  String(body.width));
      if (body.height != null)  args.push("--height", String(body.height));
      if (body.steps != null)   args.push("--steps",  String(body.steps));

      const data: ParallaxJobData = {
        action:     "create",
        media:      "image",
        model:      body.model,
        script:     getScript("create", "image", body.model) ?? "",
        args,
        scriptBase: getScriptBase(),
        uvPath:     getUvPath(),
      };

      const job_id = await submitJob(data);
      return { job_id, status: "queued" as const };
    },
    {
      body: t.Object({
        model:           t.String(),
        prompt:          t.String(),
        negative_prompt: t.Optional(t.String()),
        width:           t.Optional(t.Number()),
        height:          t.Optional(t.Number()),
        steps:           t.Optional(t.Number()),
      }),
    },
  )

  // AC02: POST /jobs/create/video
  .post(
    "/jobs/create/video",
    async ({ body }) => {
      const cfg = getModelConfig("video", body.model);
      const useI2v = body.input !== undefined && cfg?.i2v !== undefined;
      const script = cfg ? (useI2v ? cfg.i2v! : cfg.t2v) : "";

      const args: string[] = ["--prompt", body.prompt, "--output", "output.mp4"];
      if (body.input)           args.push("--image",  body.input);
      if (body.negative_prompt) args.push("--negative-prompt", body.negative_prompt);
      if (body.width != null)   args.push("--width",  String(body.width));
      if (body.height != null)  args.push("--height", String(body.height));
      if (body.frames != null)  args.push("--length", String(body.frames));
      if (body.steps != null)   args.push("--steps",  String(body.steps));

      const data: ParallaxJobData = {
        action:     "create",
        media:      "video",
        model:      body.model,
        script,
        args,
        scriptBase: getScriptBase(),
        uvPath:     getUvPath(),
      };

      const job_id = await submitJob(data);
      return { job_id, status: "queued" as const };
    },
    {
      body: t.Object({
        model:           t.String(),
        prompt:          t.String(),
        input:           t.Optional(t.String()),
        negative_prompt: t.Optional(t.String()),
        width:           t.Optional(t.Number()),
        height:          t.Optional(t.Number()),
        frames:          t.Optional(t.Number()),
        fps:             t.Optional(t.Number()),
        steps:           t.Optional(t.Number()),
      }),
    },
  )

  // AC03: POST /jobs/create/audio
  .post(
    "/jobs/create/audio",
    async ({ body }) => {
      const args: string[] = ["--tags", body.prompt, "--output", "output.wav"];
      if (body.duration_seconds != null) args.push("--duration", String(body.duration_seconds));
      if (body.steps != null)            args.push("--steps",    String(body.steps));

      const data: ParallaxJobData = {
        action:     "create",
        media:      "audio",
        model:      body.model,
        script:     getScript("create", "audio", body.model) ?? "",
        args,
        scriptBase: getScriptBase(),
        uvPath:     getUvPath(),
      };

      const job_id = await submitJob(data);
      return { job_id, status: "queued" as const };
    },
    {
      body: t.Object({
        model:            t.String(),
        prompt:           t.String(),
        negative_prompt:  t.Optional(t.String()),
        duration_seconds: t.Optional(t.Number()),
        steps:            t.Optional(t.Number()),
      }),
    },
  )

  // AC04: POST /jobs/edit/image
  .post(
    "/jobs/edit/image",
    async ({ body }) => {
      const args: string[] = [
        "--prompt", body.prompt,
        "--image",  body.image_path,
        "--output", "output.png",
      ];
      if (body.steps != null) args.push("--steps", String(body.steps));

      const data: ParallaxJobData = {
        action:     "edit",
        media:      "image",
        model:      body.model,
        script:     getScript("edit", "image", body.model) ?? "",
        args,
        scriptBase: getScriptBase(),
        uvPath:     getUvPath(),
      };

      const job_id = await submitJob(data);
      return { job_id, status: "queued" as const };
    },
    {
      body: t.Object({
        model:      t.String(),
        image_path: t.String(),
        prompt:     t.String(),
        steps:      t.Optional(t.Number()),
      }),
    },
  )

  // AC05: POST /jobs/upscale/image
  .post(
    "/jobs/upscale/image",
    async ({ body }) => {
      const output = body.output ?? "output.png";
      const args: string[] = ["--input", body.image_path, "--output", output];

      const data: ParallaxJobData = {
        action:     "upscale",
        media:      "image",
        model:      body.model,
        script:     getScript("upscale", "image", body.model) ?? "",
        args,
        scriptBase: getScriptBase(),
        uvPath:     getUvPath(),
      };

      const job_id = await submitJob(data);
      return { job_id, status: "queued" as const };
    },
    {
      body: t.Object({
        image_path: t.String(),
        model:      t.String(),
        output:     t.Optional(t.String()),
      }),
    },
  );

if (import.meta.main) {
  // Spawn the serial inference worker daemon (one job at a time).
  const workerScript = join(import.meta.dir, "../../parallax_cli/src/_worker.ts");
  const worker = Bun.spawn(["bun", workerScript], {
    stdin: "ignore",
    stdout: Bun.file("/tmp/parallax-worker.log"),
    stderr: Bun.file("/tmp/parallax-worker.err"),
  });
  worker.unref();

  app.listen(Number(process.env.PORT ?? 3000));
  console.log(`parallax_ms running at ${app.server?.hostname}:${app.server?.port}`);
}
