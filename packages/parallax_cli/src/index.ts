#!/usr/bin/env bun
import { Command } from "commander";

// Known models per action+media — single source of truth for help text and validation
const MODELS: Record<string, string[]> = {
  "create image": ["sdxl", "anima", "z_image", "flux_klein", "qwen"],
  "create video": ["ltx2", "ltx23", "wan21", "wan22"],
  "create audio": ["ace_step"],
  "edit image": ["qwen"],
  "edit video": ["wan21", "wan22"],
};

function modelsFooter(key: string): string {
  return `\nAvailable models: ${MODELS[key].join(", ")}`;
}

function validateModel(key: string, model: string): void {
  if (!MODELS[key].includes(model)) {
    console.error(
      `Error: unknown model "${model}" for ${key}. Known models: ${MODELS[key].join(", ")}`
    );
    process.exit(1);
  }
}

const NOT_IMPLEMENTED = (): never => {
  console.error("Not yet implemented — coming soon.");
  process.exit(1);
};

const program = new Command();

program
  .name("parallax")
  .description("Parallax CLI — comfy-diffusion media generation")
  .version("0.1.0")
  .addHelpText("before", "parallax v0.1.0\n");

// ── create ────────────────────────────────────────────────────────────────────

const create = program
  .command("create")
  .description("Generate media from a text prompt")
  .usage("<media> [options]");

create
  .command("image")
  .description("Generate an image from a text prompt")
  .requiredOption("--model <name>", `Model to use (choices: ${MODELS["create image"].join(", ")})`)
  .requiredOption("--prompt <text>", "Text prompt describing the image to generate")
  .option("--negative-prompt <text>", "Negative prompt (what to avoid in the output)")
  .option("--width <pixels>", "Image width in pixels", "1024")
  .option("--height <pixels>", "Image height in pixels", "1024")
  .option("--steps <n>", "Number of sampling steps", "20")
  .option("--cfg <value>", "CFG guidance scale", "7")
  .option("--seed <n>", "Random seed for reproducibility")
  .option("--output <path>", "Output file path", "output.png")
  .addHelpText("after", modelsFooter("create image"))
  .action((opts) => { validateModel("create image", opts.model); NOT_IMPLEMENTED(); });

create
  .command("video")
  .description("Generate a video from a text prompt")
  .requiredOption("--model <name>", `Model to use (choices: ${MODELS["create video"].join(", ")})`)
  .requiredOption("--prompt <text>", "Text prompt describing the video to generate")
  .option("--width <pixels>", "Video width in pixels", "832")
  .option("--height <pixels>", "Video height in pixels", "480")
  .option("--length <frames>", "Number of frames to generate", "81")
  .option("--steps <n>", "Number of sampling steps", "30")
  .option("--cfg <value>", "CFG guidance scale", "6")
  .option("--seed <n>", "Random seed for reproducibility")
  .option("--output <path>", "Output file path", "output.mp4")
  .addHelpText("after", modelsFooter("create video"))
  .action((opts) => { validateModel("create video", opts.model); NOT_IMPLEMENTED(); });

create
  .command("audio")
  .description("Generate audio from a text prompt")
  .requiredOption("--model <name>", `Model to use (choices: ${MODELS["create audio"].join(", ")})`)
  .requiredOption("--prompt <text>", "Text prompt describing the audio to generate")
  .option("--length <seconds>", "Duration in seconds", "30")
  .option("--steps <n>", "Number of sampling steps", "60")
  .option("--seed <n>", "Random seed for reproducibility")
  .option("--output <path>", "Output file path", "output.wav")
  .addHelpText("after", modelsFooter("create audio"))
  .action((opts) => { validateModel("create audio", opts.model); NOT_IMPLEMENTED(); });

// ── edit ──────────────────────────────────────────────────────────────────────

const edit = program
  .command("edit")
  .description("Edit existing media using a text prompt")
  .usage("<media> [options]");

edit
  .command("image")
  .description("Edit an image using a text prompt")
  .requiredOption("--model <name>", `Model to use (choices: ${MODELS["edit image"].join(", ")})`)
  .requiredOption("--prompt <text>", "Text prompt describing the desired edits")
  .requiredOption("--input <path>", "Path to the input image file")
  .option("--steps <n>", "Number of sampling steps", "20")
  .option("--cfg <value>", "CFG guidance scale", "7")
  .option("--seed <n>", "Random seed for reproducibility")
  .option("--output <path>", "Output file path", "output.png")
  .addHelpText("after", modelsFooter("edit image"))
  .action((opts) => { validateModel("edit image", opts.model); NOT_IMPLEMENTED(); });

edit
  .command("video")
  .description("Edit a video using a text prompt")
  .requiredOption("--model <name>", `Model to use (choices: ${MODELS["edit video"].join(", ")})`)
  .requiredOption("--prompt <text>", "Text prompt describing the desired edits")
  .requiredOption("--input <path>", "Path to the input video file")
  .option("--width <pixels>", "Output video width in pixels")
  .option("--height <pixels>", "Output video height in pixels")
  .option("--length <frames>", "Number of frames to generate")
  .option("--steps <n>", "Number of sampling steps", "30")
  .option("--cfg <value>", "CFG guidance scale", "6")
  .option("--seed <n>", "Random seed for reproducibility")
  .option("--output <path>", "Output file path", "output.mp4")
  .addHelpText("after", modelsFooter("edit video"))
  .action((opts) => { validateModel("edit video", opts.model); NOT_IMPLEMENTED(); });

if (process.argv.length <= 2) {
  program.help();
}

program.parse();
