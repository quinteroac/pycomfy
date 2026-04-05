import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import { submitJob } from "@parallax/sdk/submit";
import { getJobStatus, getQueue } from "@parallax/sdk";
import type { ParallaxJobData } from "@parallax/sdk";
import { getScript, getModels, getModelConfig } from "@parallax/cli/src/models/registry";

const server = new McpServer({
  name: "parallax-mcp",
  version: "0.1.0",
});

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
    description: "Generate an image using the Parallax pipeline (parallax create image). Returns a job_id. Use get_job_status to poll or wait_for_job to block until done.",
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
    const validModels = getModels("create", "image");
    if (!validModels.includes(input.model)) {
      return {
        isError: true,
        content: [{ type: "text", text: `Unknown model '${input.model}'. Valid models: ${validModels.join(", ")}` }],
      };
    }

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
      script:     getScript("create", "image", input.model) ?? "",
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
    description: "Generate a video using the Parallax pipeline (parallax create video). Returns a job_id. Use get_job_status to poll or wait_for_job to block until done.",
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
    const validModels = getModels("create", "video");
    if (!validModels.includes(input.model)) {
      return {
        isError: true,
        content: [{ type: "text", text: `Unknown model '${input.model}'. Valid models: ${validModels.join(", ")}` }],
      };
    }

    const modelsDir = getModelsDir(input.modelsDir);
    const modelConfig = getModelConfig("video", input.model);
    const useI2v = input.input !== undefined && modelConfig?.i2v !== undefined;
    const script = modelConfig ? (useI2v ? modelConfig.i2v! : modelConfig.t2v) : "";

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
    description: "Generate audio using the Parallax pipeline (parallax create audio). Returns a job_id. Use get_job_status to poll or wait_for_job to block until done.",
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
    const validModels = getModels("create", "audio");
    if (!validModels.includes(input.model)) {
      return {
        isError: true,
        content: [{ type: "text", text: `Unknown model '${input.model}'. Valid models: ${validModels.join(", ")}` }],
      };
    }

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
      script:     getScript("create", "audio", input.model) ?? "",
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
    description: "Edit an image using the Parallax pipeline (parallax edit image). Returns a job_id. Use get_job_status to poll or wait_for_job to block until done.",
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
    const validModels = getModels("edit", "image");
    if (!validModels.includes(input.model)) {
      return {
        isError: true,
        content: [{ type: "text", text: `Unknown model '${input.model}'. Valid models: ${validModels.join(", ")}` }],
      };
    }

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
      script:     getScript("edit", "image", input.model) ?? "",
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
    description: "Upscale an image using the Parallax pipeline (parallax upscale image). Returns a job_id. Use get_job_status to poll or wait_for_job to block until done.",
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
    const validModels = getModels("upscale", "image");
    if (!validModels.includes(input.model)) {
      return {
        isError: true,
        content: [{ type: "text", text: `Unknown model '${input.model}'. Valid models: ${validModels.join(", ")}` }],
      };
    }

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
      script:     getScript("upscale", "image", input.model) ?? "",
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
    description: "Check the current status and progress of a submitted inference job. Returns status, progress percentage (0-100), and output path when completed.",
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
    description: "Block until a submitted inference job completes. Polls internally every 2 seconds. Default timeout: 600 seconds. Returns output path on success.",
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
