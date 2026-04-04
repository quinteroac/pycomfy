import { Elysia, t } from "elysia";
import { Stream } from "@elysiajs/stream";
import { submitJob, getJobStatus, listJobs, cancelJob, getQueueStats } from "@parallax/sdk";
import type { JobStatusValue } from "@parallax/sdk";
import type { ParallaxJobData } from "@parallax/sdk";

// Script registry — mirrors parallax_cli/src/models/registry.ts
const IMAGE_CREATE_SCRIPTS: Record<string, string> = {
  sdxl:       "runtime/image/generation/sdxl/t2i.py",
  anima:      "runtime/image/generation/anima/t2i.py",
  z_image:    "runtime/image/generation/z_image/turbo.py",
  flux_klein: "runtime/image/generation/flux/4b_distilled.py",
  qwen:       "runtime/image/generation/qwen/layered_t2l.py",
};

const IMAGE_EDIT_SCRIPTS: Record<string, string> = {
  flux_4b_base:      "runtime/image/edit/flux/4b_base.py",
  flux_4b_distilled: "runtime/image/edit/flux/4b_distilled.py",
  flux_9b_base:      "runtime/image/edit/flux/9b_base.py",
  flux_9b_distilled: "runtime/image/edit/flux/9b_distilled.py",
  flux_9b_kv:        "runtime/image/edit/flux/9b_kv.py",
  qwen:              "runtime/image/edit/qwen/edit_2511.py",
};

const IMAGE_UPSCALE_SCRIPTS: Record<string, string> = {
  esrgan:         "runtime/image/edit/sd/esrgan_upscale.py",
  latent_upscale: "runtime/image/edit/sd/latent_upscale.py",
};

const VIDEO_CREATE_SCRIPTS: Record<string, { t2v: string; i2v?: string }> = {
  ltx2:  { t2v: "runtime/video/ltx/ltx2/t2v.py",  i2v: "runtime/video/ltx/ltx2/i2v.py"  },
  ltx23: { t2v: "runtime/video/ltx/ltx23/t2v.py", i2v: "runtime/video/ltx/ltx23/i2v.py" },
  wan21: { t2v: "runtime/video/wan/wan21/t2v.py",  i2v: "runtime/video/wan/wan21/i2v.py"  },
  wan22: { t2v: "runtime/video/wan/wan22/t2v.py",  i2v: "runtime/video/wan/wan22/i2v.py"  },
};

const AUDIO_CREATE_SCRIPTS: Record<string, string> = {
  ace_step: "runtime/audio/ace/t2a.py",
};

function getScriptBase(): string {
  return process.env.PARALLAX_RUNTIME_DIR ?? process.env.PARALLAX_REPO_ROOT ?? process.cwd();
}

function getUvPath(): string {
  return process.env.PARALLAX_UV_PATH ?? "uv";
}

export const app = new Elysia()
  .get("/health", async () => {
    const queue = await getQueueStats();
    return { status: "ok", queue };
  })
  .onError(({ code, error, set }) => {
    if (code === "VALIDATION") {
      set.status = 400;
      return { error: error.message };
    }
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
    if (result === "terminal") {
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
        script:     IMAGE_CREATE_SCRIPTS[body.model] ?? "",
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
      const cfg = VIDEO_CREATE_SCRIPTS[body.model];
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
        script:     AUDIO_CREATE_SCRIPTS[body.model] ?? "",
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
        script:     IMAGE_EDIT_SCRIPTS[body.model] ?? "",
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
        script:     IMAGE_UPSCALE_SCRIPTS[body.model] ?? "",
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
  app.listen(3000);
  console.log(`parallax_ms running at ${app.server?.hostname}:${app.server?.port}`);
}
