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

const transport = new StdioServerTransport();
await server.connect(transport);
