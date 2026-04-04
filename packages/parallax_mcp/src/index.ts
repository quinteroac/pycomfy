import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import { resolve, join } from "path";

const server = new McpServer({
  name: "parallax-mcp",
  version: "0.1.0",
});

// Absolute path to the @parallax/cli package directory.
const CLI_DIR = resolve(join(import.meta.dir, "../../parallax_cli"));

server.registerTool(
  "create_image",
  {
    description: "Generate an image using the Parallax pipeline (parallax create image)",
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
    const args: string[] = ["create", "image", "--model", input.model, "--prompt", input.prompt];
    if (input.negativePrompt) args.push("--negative-prompt", input.negativePrompt);
    if (input.width)          args.push("--width",           input.width);
    if (input.height)         args.push("--height",          input.height);
    if (input.steps)          args.push("--steps",           input.steps);
    if (input.cfg)            args.push("--cfg",             input.cfg);
    if (input.seed)           args.push("--seed",            input.seed);
    if (input.output)         args.push("--output",          input.output);
    if (input.modelsDir)      args.push("--models-dir",      input.modelsDir);

    const outputPath = resolve(input.output ?? "output.png");

    const proc = Bun.spawn(["bun", "run", "src/index.ts", ...args], {
      stdout: "pipe",
      stderr: "pipe",
      cwd: CLI_DIR,
    });

    const [exitCode, stderr] = await Promise.all([
      proc.exited,
      new Response(proc.stderr).text(),
    ]);

    if (exitCode !== 0) {
      return {
        content: [{ type: "text", text: `Error: ${stderr.trim()}` }],
        isError: true,
      };
    }

    return {
      content: [{ type: "text", text: outputPath }],
    };
  },
);

server.registerTool(
  "create_video",
  {
    description: "Generate a video using the Parallax pipeline (parallax create video)",
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
    const args: string[] = ["create", "video", "--model", input.model, "--prompt", input.prompt];
    if (input.input)     args.push("--input",      input.input);
    if (input.width)     args.push("--width",      input.width);
    if (input.height)    args.push("--height",     input.height);
    if (input.length)    args.push("--length",     input.length);
    if (input.steps)     args.push("--steps",      input.steps);
    if (input.cfg)       args.push("--cfg",        input.cfg);
    if (input.seed)      args.push("--seed",       input.seed);
    if (input.output)    args.push("--output",     input.output);
    if (input.modelsDir) args.push("--models-dir", input.modelsDir);

    const outputPath = resolve(input.output ?? "output.mp4");

    const proc = Bun.spawn(["bun", "run", "src/index.ts", ...args], {
      stdout: "pipe",
      stderr: "pipe",
      cwd: CLI_DIR,
    });

    const [exitCode, stderr] = await Promise.all([
      proc.exited,
      new Response(proc.stderr).text(),
    ]);

    if (exitCode !== 0) {
      return {
        content: [{ type: "text", text: `Error: ${stderr.trim()}` }],
        isError: true,
      };
    }

    return {
      content: [{ type: "text", text: outputPath }],
    };
  },
);

server.registerTool(
  "create_audio",
  {
    description: "Generate audio using the Parallax pipeline (parallax create audio)",
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
    const args: string[] = ["create", "audio", "--model", input.model, "--prompt", input.prompt];
    if (input.length)    args.push("--length",     input.length);
    if (input.steps)     args.push("--steps",      input.steps);
    if (input.cfg)       args.push("--cfg",        input.cfg);
    if (input.bpm)       args.push("--bpm",        input.bpm);
    if (input.lyrics)    args.push("--lyrics",     input.lyrics);
    if (input.seed)      args.push("--seed",       input.seed);
    if (input.output)    args.push("--output",     input.output);
    if (input.modelsDir) args.push("--models-dir", input.modelsDir);

    const outputPath = resolve(input.output ?? "output.wav");

    const proc = Bun.spawn(["bun", "run", "src/index.ts", ...args], {
      stdout: "pipe",
      stderr: "pipe",
      cwd: CLI_DIR,
    });

    const [exitCode, stderr] = await Promise.all([
      proc.exited,
      new Response(proc.stderr).text(),
    ]);

    if (exitCode !== 0) {
      return {
        content: [{ type: "text", text: `Error: ${stderr.trim()}` }],
        isError: true,
      };
    }

    return {
      content: [{ type: "text", text: outputPath }],
    };
  },
);

server.registerTool(
  "edit_image",
  {
    description: "Edit an image using the Parallax pipeline (parallax edit image)",
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
    const args: string[] = ["edit", "image", "--model", input.model, "--prompt", input.prompt, "--input", input.input];
    if (input.subjectImage) args.push("--subject-image", input.subjectImage);
    if (input.width)        args.push("--width",         input.width);
    if (input.height)       args.push("--height",        input.height);
    if (input.steps)        args.push("--steps",         input.steps);
    if (input.cfg)          args.push("--cfg",           input.cfg);
    if (input.seed)         args.push("--seed",          input.seed);
    if (input.output)       args.push("--output",        input.output);
    if (input.image2)       args.push("--image2",        input.image2);
    if (input.image3)       args.push("--image3",        input.image3);
    if (input.noLora)       args.push("--no-lora");
    if (input.modelsDir)    args.push("--models-dir",    input.modelsDir);

    const outputPath = resolve(input.output ?? "output.png");

    const proc = Bun.spawn(["bun", "run", "src/index.ts", ...args], {
      stdout: "pipe",
      stderr: "pipe",
      cwd: CLI_DIR,
    });

    const [exitCode, stderr] = await Promise.all([
      proc.exited,
      new Response(proc.stderr).text(),
    ]);

    if (exitCode !== 0) {
      return {
        content: [{ type: "text", text: `Error: ${stderr.trim()}` }],
        isError: true,
      };
    }

    return {
      content: [{ type: "text", text: outputPath }],
    };
  },
);

server.registerTool(
  "upscale_image",
  {
    description: "Upscale an image using the Parallax pipeline (parallax upscale image)",
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
    const args: string[] = ["upscale", "image", "--model", input.model, "--prompt", input.prompt, "--input", input.input];
    if (input.checkpoint)              args.push("--checkpoint",               input.checkpoint);
    if (input.esrganCheckpoint)        args.push("--esrgan-checkpoint",        input.esrganCheckpoint);
    if (input.latentUpscaleCheckpoint) args.push("--latent-upscale-checkpoint", input.latentUpscaleCheckpoint);
    if (input.negativePrompt)          args.push("--negative-prompt",          input.negativePrompt);
    if (input.width)                   args.push("--width",                    input.width);
    if (input.height)                  args.push("--height",                   input.height);
    if (input.steps)                   args.push("--steps",                    input.steps);
    if (input.cfg)                     args.push("--cfg",                      input.cfg);
    if (input.seed)                    args.push("--seed",                     input.seed);
    if (input.output)                  args.push("--output",                   input.output);
    if (input.outputBase)              args.push("--output-base",              input.outputBase);
    if (input.modelsDir)               args.push("--models-dir",               input.modelsDir);

    const outputPath = resolve(input.output ?? "output.png");

    const proc = Bun.spawn(["bun", "run", "src/index.ts", ...args], {
      stdout: "pipe",
      stderr: "pipe",
      cwd: CLI_DIR,
    });

    const [exitCode, stderr] = await Promise.all([
      proc.exited,
      new Response(proc.stderr).text(),
    ]);

    if (exitCode !== 0) {
      return {
        content: [{ type: "text", text: `Error: ${stderr.trim()}` }],
        isError: true,
      };
    }

    return {
      content: [{ type: "text", text: outputPath }],
    };
  },
);

const transport = new StdioServerTransport();
await server.connect(transport);
