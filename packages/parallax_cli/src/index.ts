#!/usr/bin/env bun
import { Command } from "commander";
import { getModels } from "./models/registry";
import { setupCreateCommand } from "./commands/create";

function modelsFooter(action: string, media: string): string {
  return `\nAvailable models: ${getModels(action, media).join(", ")}`;
}

function validateModel(action: string, media: string, model: string): void {
  const known = getModels(action, media);
  if (!known.includes(model)) {
    console.error(
      `Error: unknown model "${model}" for ${action} ${media}. Known models: ${known.join(", ")}`
    );
    process.exit(1);
  }
}

function notImplemented(action: string, media: string, model: string): never {
  console.log(`[parallax] ${action} ${media} --model ${model} — not yet implemented (coming soon)`);
  process.exit(0);
}

const program = new Command();

// Reformat commander's missing-required-option errors to "Error: --flag is required"
function formatRequiredFlagError(msg: string): string {
  return msg.replace(
    /error: required option '(--[a-z-]+)[^']*' not specified/,
    "Error: $1 is required"
  );
}

program
  .name("parallax")
  .description("Parallax CLI — comfy-diffusion media generation")
  .version("0.1.0")
  .addHelpText("before", "parallax v0.1.0\n")
  .configureOutput({
    writeErr: (str) => process.stderr.write(formatRequiredFlagError(str)),
  });

// ── create ────────────────────────────────────────────────────────────────────

setupCreateCommand(program);

// ── edit ──────────────────────────────────────────────────────────────────────

const edit = program
  .command("edit")
  .description("Edit existing media using a text prompt")
  .usage("<media> [options]");

edit
  .command("image")
  .description("Edit an image using a text prompt")
  .requiredOption("--model <name>", `Model to use (choices: ${getModels("edit", "image").join(", ")})`)
  .requiredOption("--prompt <text>", "Text prompt describing the desired edits")
  .requiredOption("--input <path>", "Path to the input image file")
  .option("--steps <n>", "Number of sampling steps", "20")
  .option("--cfg <value>", "CFG guidance scale", "7")
  .option("--seed <n>", "Random seed for reproducibility")
  .option("--output <path>", "Output file path", "output.png")
  .addHelpText("after", modelsFooter("edit", "image"))
  .action((opts) => { validateModel("edit", "image", opts.model); notImplemented("edit", "image", opts.model); });

edit
  .command("video")
  .description("Edit a video using a text prompt")
  .requiredOption("--model <name>", `Model to use (choices: ${getModels("edit", "video").join(", ")})`)
  .requiredOption("--prompt <text>", "Text prompt describing the desired edits")
  .requiredOption("--input <path>", "Path to the input video file")
  .option("--width <pixels>", "Output video width in pixels")
  .option("--height <pixels>", "Output video height in pixels")
  .option("--length <frames>", "Number of frames to generate")
  .option("--steps <n>", "Number of sampling steps", "30")
  .option("--cfg <value>", "CFG guidance scale", "6")
  .option("--seed <n>", "Random seed for reproducibility")
  .option("--output <path>", "Output file path", "output.mp4")
  .addHelpText("after", modelsFooter("edit", "video"))
  .action((opts) => { validateModel("edit", "video", opts.model); notImplemented("edit", "video", opts.model); });

if (process.argv.length <= 2) {
  program.help();
}

program.parse();
