import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import { submitJob } from "@parallax/sdk/submit";
import { getJobStatus, getQueue } from "@parallax/sdk";
import type { ParallaxJobData } from "@parallax/sdk";

const server = new McpServer({
  name: "parallax-mcp",
  version: "0.1.0",
});

// Script registries — mirrors parallax_cli/src/models/registry.ts
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

function getModelsDir(override?: string): string {
  return override ?? process.env.PYCOMFY_MODELS_DIR ?? "";
}

server.registerTool(
  "create_image",
  {
    description: "Generate an image using the Parallax pipeline (parallax create image). Returns a job ID immediately — use get_job_status to poll for the output path.",
    inputSchema: {
      model:          z.string().describe("Model to use (e.g. sdxl, anima, z_image, flux_klein)"),
      prompt:         z.string().describe("Text prompt describing the image to generate"),
      negativePrompt: z.string().optional().describe("Negative prompt (what to avoid)"),
      width:          z.string().optional().describe("Image width in pixels"),
      height:         z.string().optional().describe("Image height in pixels"),
      steps:          z.string().optional().describe("Number of sampling steps"),
      cfg:            z.string().optional().describe("CFG guidance scale"),
      seed:           z.string().optional().describe("Random seed for reproducibility"),
      output:         z.string().optional().describe("Output file path (default: output.png)"),
      modelsDir:      z.string().optional().describe("Models directory (overrides PYCOMFY_MODELS_DIR)"),
    },
  },
  async (input) => {
    const modelsDir = getModelsDir(input.modelsDir);
    const args: string[] = ["--models-dir", modelsDir, "--prompt", input.prompt];
    if (input.model !== "z_image") {
      if (input.negativePrompt) args.push("--negative-prompt", input.negativePrompt);
    }
    if (input.width)  args.push("--width",  input.width);
    if (input.height) args.push("--height", input.height);
    if (input.steps)  args.push("--steps",  input.steps);
    if (input.model !== "z_image" && input.cfg) args.push("--cfg", input.cfg);
    if (input.seed)   args.push("--seed",   input.seed);
    args.push("--output", input.output ?? "output.png");

    const data: ParallaxJobData = {
      action:     "create",
      media:      "image",
      model:      input.model,
      script:     IMAGE_CREATE_SCRIPTS[input.model] ?? "",
      args,
      scriptBase: getScriptBase(),
      uvPath:     getUvPath(),
    };

    const jobId = await submitJob(data);
    return {
      content: [{ type: "text", text: `job_id: ${jobId}\nstatus: queued\nmodel: ${input.model}` }],
    };
  },
);

server.registerTool(
  "create_video",
  {
    description: "Generate a video using the Parallax pipeline (parallax create video). Returns a job ID immediately — use get_job_status to poll for the output path.",
    inputSchema: {
      model:     z.string().describe("Model to use (e.g. ltx2, ltx23, wan21, wan22)"),
      prompt:    z.string().describe("Text prompt describing the video to generate"),
      input:     z.string().optional().describe("Input image path for image-to-video (ltx2, ltx23, wan21, wan22)"),
      width:     z.string().optional().describe("Video width in pixels"),
      height:    z.string().optional().describe("Video height in pixels"),
      length:    z.string().optional().describe("Number of frames to generate"),
      steps:     z.string().optional().describe("Number of sampling steps"),
      cfg:       z.string().optional().describe("CFG guidance scale"),
      seed:      z.string().optional().describe("Random seed for reproducibility"),
      output:    z.string().optional().describe("Output file path (default: output.mp4)"),
      modelsDir: z.string().optional().describe("Models directory (overrides PYCOMFY_MODELS_DIR)"),
    },
  },
  async (input) => {
    const modelsDir = getModelsDir(input.modelsDir);
    const cfg = VIDEO_CREATE_SCRIPTS[input.model];
    const useI2v = input.input !== undefined && cfg?.i2v !== undefined;
    const script = cfg ? (useI2v ? cfg.i2v! : cfg.t2v) : "";

    const args: string[] = ["--models-dir", modelsDir, "--prompt", input.prompt];
    if (useI2v) args.push("--image", input.input!);
    if (input.width)  args.push("--width",  input.width);
    if (input.height) args.push("--height", input.height);
    if (input.length) args.push("--length", input.length);
    if (input.steps)  args.push("--steps",  input.steps);
    if (input.cfg)    args.push("--cfg",    input.cfg);
    if (input.seed)   args.push("--seed",   input.seed);
    args.push("--output", input.output ?? "output.mp4");

    const data: ParallaxJobData = {
      action:     "create",
      media:      "video",
      model:      input.model,
      script,
      args,
      scriptBase: getScriptBase(),
      uvPath:     getUvPath(),
    };

    const jobId = await submitJob(data);
    return {
      content: [{ type: "text", text: `job_id: ${jobId}\nstatus: queued\nmodel: ${input.model}` }],
    };
  },
);

server.registerTool(
  "create_audio",
  {
    description: "Generate audio using the Parallax pipeline (parallax create audio). Returns a job ID immediately — use get_job_status to poll for the output path.",
    inputSchema: {
      model:     z.string().describe("Model to use (e.g. ace_step)"),
      prompt:    z.string().describe("Text prompt describing the audio to generate"),
      length:    z.string().optional().describe("Duration in seconds"),
      steps:     z.string().optional().describe("Number of sampling steps"),
      cfg:       z.string().optional().describe("CFG guidance scale"),
      bpm:       z.string().optional().describe("Beats per minute"),
      lyrics:    z.string().optional().describe("Lyrics text (ace_step)"),
      seed:      z.string().optional().describe("Random seed for reproducibility"),
      output:    z.string().optional().describe("Output file path (default: output.wav)"),
      modelsDir: z.string().optional().describe("Models directory (overrides PYCOMFY_MODELS_DIR)"),
    },
  },
  async (input) => {
    const modelsDir = getModelsDir(input.modelsDir);
    const args: string[] = ["--models-dir", modelsDir, "--tags", input.prompt];
    if (input.length) args.push("--duration", input.length);
    if (input.steps)  args.push("--steps",    input.steps);
    if (input.cfg)    args.push("--cfg",       input.cfg);
    if (input.bpm)    args.push("--bpm",       input.bpm);
    if (input.lyrics !== undefined) args.push("--lyrics", input.lyrics);
    if (input.seed)   args.push("--seed",      input.seed);
    args.push("--output", input.output ?? "output.wav");

    const data: ParallaxJobData = {
      action:     "create",
      media:      "audio",
      model:      input.model,
      script:     AUDIO_CREATE_SCRIPTS[input.model] ?? "",
      args,
      scriptBase: getScriptBase(),
      uvPath:     getUvPath(),
    };

    const jobId = await submitJob(data);
    return {
      content: [{ type: "text", text: `job_id: ${jobId}\nstatus: queued\nmodel: ${input.model}` }],
    };
  },
);

server.registerTool(
  "edit_image",
  {
    description: "Edit an image using the Parallax pipeline (parallax edit image). Returns a job ID immediately — use get_job_status to poll for the output path.",
    inputSchema: {
      model:        z.string().describe("Model to use (e.g. flux_klein, qwen)"),
      prompt:       z.string().describe("Text prompt describing the desired edits"),
      input:        z.string().describe("Path to the input image file"),
      subjectImage: z.string().optional().describe("Subject reference image (flux_9b_kv only)"),
      width:        z.string().optional().describe("Image width in pixels"),
      height:       z.string().optional().describe("Image height in pixels"),
      steps:        z.string().optional().describe("Number of sampling steps"),
      cfg:          z.string().optional().describe("CFG guidance scale"),
      seed:         z.string().optional().describe("Random seed for reproducibility"),
      output:       z.string().optional().describe("Output file path (default: output.png)"),
      image2:       z.string().optional().describe("Second input image (qwen only)"),
      image3:       z.string().optional().describe("Third input image (qwen only)"),
      noLora:       z.boolean().optional().describe("Disable LoRA (qwen only)"),
      modelsDir:    z.string().optional().describe("Models directory (overrides PYCOMFY_MODELS_DIR)"),
    },
  },
  async (input) => {
    const modelsDir = getModelsDir(input.modelsDir);
    const outputPath = input.output ?? "output.png";
    const args: string[] = ["--models-dir", modelsDir];

    if (input.model === "qwen") {
      args.push("--image", input.input, "--prompt", input.prompt);
      if (input.steps !== undefined) args.push("--steps", input.steps);
      if (input.cfg !== undefined)   args.push("--cfg",   input.cfg);
      if (input.seed !== undefined)  args.push("--seed",  input.seed);
      const prefix = outputPath.endsWith(".png") ? outputPath.slice(0, -4) : outputPath;
      args.push("--output-prefix", prefix);
      if (input.image2 !== undefined) args.push("--image2", input.image2);
      if (input.image3 !== undefined) args.push("--image3", input.image3);
      if (input.noLora) args.push("--no-lora");
    } else {
      args.push("--prompt", input.prompt, "--image", input.input);
      if (input.width  !== undefined) args.push("--width",  input.width);
      if (input.height !== undefined) args.push("--height", input.height);
      if (input.steps  !== undefined) args.push("--steps",  input.steps);
      if (input.seed   !== undefined) args.push("--seed",   input.seed);
      args.push("--output", outputPath);
      if (input.model === "flux_9b_kv" && input.subjectImage !== undefined) {
        args.push("--subject-image", input.subjectImage);
      }
    }

    const data: ParallaxJobData = {
      action:     "edit",
      media:      "image",
      model:      input.model,
      script:     IMAGE_EDIT_SCRIPTS[input.model] ?? "",
      args,
      scriptBase: getScriptBase(),
      uvPath:     getUvPath(),
    };

    const jobId = await submitJob(data);
    return {
      content: [{ type: "text", text: `job_id: ${jobId}\nstatus: queued\nmodel: ${input.model}` }],
    };
  },
);

server.registerTool(
  "upscale_image",
  {
    description: "Upscale an image using the Parallax pipeline (parallax upscale image). Returns a job ID immediately — use get_job_status to poll for the output path.",
    inputSchema: {
      model:                   z.string().describe("Model to use (e.g. esrgan, latent_upscale)"),
      prompt:                  z.string().describe("Text prompt"),
      input:                   z.string().describe("Path to the input image file to upscale"),
      checkpoint:              z.string().optional().describe("Base checkpoint filename (overrides PYCOMFY_CHECKPOINT)"),
      esrganCheckpoint:        z.string().optional().describe("ESRGAN checkpoint filename (required for esrgan)"),
      latentUpscaleCheckpoint: z.string().optional().describe("Latent upscale checkpoint filename (required for latent_upscale)"),
      negativePrompt:          z.string().optional().describe("Negative prompt (what to avoid)"),
      width:                   z.string().optional().describe("Image width in pixels"),
      height:                  z.string().optional().describe("Image height in pixels"),
      steps:                   z.string().optional().describe("Number of sampling steps"),
      cfg:                     z.string().optional().describe("CFG guidance scale"),
      seed:                    z.string().optional().describe("Random seed for reproducibility"),
      output:                  z.string().optional().describe("Output file path (default: output.png)"),
      outputBase:              z.string().optional().describe("Intermediate base image before upscaling (default: output_base.png)"),
      modelsDir:               z.string().optional().describe("Models directory (overrides PYCOMFY_MODELS_DIR)"),
    },
  },
  async (input) => {
    const modelsDir = getModelsDir(input.modelsDir);
    const args: string[] = ["--models-dir", modelsDir, "--input", input.input, "--prompt", input.prompt];
    if (input.checkpoint)              args.push("--checkpoint",               input.checkpoint);
    if (input.negativePrompt)          args.push("--negative-prompt",          input.negativePrompt);
    if (input.width)                   args.push("--width",                    input.width);
    if (input.height)                  args.push("--height",                   input.height);
    if (input.steps)                   args.push("--steps",                    input.steps);
    if (input.cfg)                     args.push("--cfg",                      input.cfg);
    if (input.seed)                    args.push("--seed",                     input.seed);
    if (input.esrganCheckpoint)        args.push("--esrgan-checkpoint",        input.esrganCheckpoint);
    if (input.latentUpscaleCheckpoint) args.push("--latent-upscale-checkpoint", input.latentUpscaleCheckpoint);
    args.push("--output",      input.output     ?? "output.png");
    args.push("--output-base", input.outputBase ?? "output_base.png");

    const data: ParallaxJobData = {
      action:     "upscale",
      media:      "image",
      model:      input.model,
      script:     IMAGE_UPSCALE_SCRIPTS[input.model] ?? "",
      args,
      scriptBase: getScriptBase(),
      uvPath:     getUvPath(),
    };

    const jobId = await submitJob(data);
    return {
      content: [{ type: "text", text: `job_id: ${jobId}\nstatus: queued\nmodel: ${input.model}` }],
    };
  },
);

server.registerTool(
  "get_job_status",
  {
    description: "Check the status of an inference job by its ID. Returns current state, progress, and output path when done.",
    inputSchema: {
      job_id: z.string().describe("The job ID returned by a create/edit/upscale tool"),
    },
  },
  async (input) => {
    const status = await getJobStatus(input.job_id);
    if (!status) {
      return {
        isError: true,
        content: [{ type: "text", text: `Job ${input.job_id} not found` }],
      };
    }
    const payload = {
      id:       status.id,
      status:   status.status,
      progress: status.progress,
      output:   status.output,
      error:    status.error,
      model:    status.model,
      action:   status.action,
      media:    status.media,
    };
    return {
      content: [{ type: "text", text: JSON.stringify(payload) }],
    };
  },
);

server.registerTool(
  "wait_for_job",
  {
    description: "Block until an inference job completes and return the output path. Polls every 2 seconds up to timeout_seconds (default 600). Returns output on success, isError on failure or timeout.",
    inputSchema: {
      job_id:          z.string().describe("The job ID to wait for"),
      timeout_seconds: z.number().optional().default(600).describe("Maximum seconds to wait (default: 600)"),
    },
  },
  async (input) => {
    const timeoutSeconds = input.timeout_seconds ?? 600;
    const deadline = Date.now() + timeoutSeconds * 1000;
    const queue = getQueue();
    const startedAt = Date.now();

    try {
      while (Date.now() < deadline) {
        const job = await queue.getJob(input.job_id);
        if (job) {
          const state = await job.getState();
          if (state === "completed") {
            const result = job.returnvalue as { outputPath?: string } | null;
            const durationSeconds = Math.round((Date.now() - startedAt) / 1000);
            return {
              content: [{
                type: "text",
                text: JSON.stringify({
                  status: "completed",
                  output: result?.outputPath ?? null,
                  duration_seconds: durationSeconds,
                }),
              }],
            };
          }
          if (state === "failed") {
            return {
              isError: true,
              content: [{
                type: "text",
                text: JSON.stringify({
                  status: "failed",
                  error: (job as any).failedReason ?? "Unknown error",
                }),
              }],
            };
          }
        }
        await new Promise((resolve) => setTimeout(resolve, 2000));
      }

      return {
        isError: true,
        content: [{
          type: "text",
          text: JSON.stringify({
            status: "timeout",
            job_id: input.job_id,
            message: `Job did not complete within ${timeoutSeconds} seconds. Use get_job_status to check later.`,
          }),
        }],
      };
    } finally {
      await queue.close();
    }
  },
);

const transport = new StdioServerTransport();
await server.connect(transport);
