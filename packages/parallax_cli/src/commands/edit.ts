// Edit command and all its subcommands (image, video).
// Action handlers follow: validate → resolveModelsDir → buildArgs → spawnPipeline.

import { Command } from "commander";
import { readConfig } from "../config";
import { getModels } from "../models/registry";

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

// Resolves --models-dir with priority: flag > stored config > env var.
function resolveModelsDir(flag?: string): string {
  const modelsDir = flag ?? readConfig().modelsDir;
  if (!modelsDir) {
    console.error("Error: --models-dir or PYCOMFY_MODELS_DIR is required");
    process.exit(1);
  }
  return modelsDir;
}

export function registerEdit(program: Command): void {
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
    .option("--models-dir <path>", "Models directory (overrides PYCOMFY_MODELS_DIR)")
    .addHelpText("after", modelsFooter("edit", "image"))
    .action((opts) => {
      validateModel("edit", "image", opts.model);
      notImplemented("edit", "image", opts.model);
    });

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
    .option("--models-dir <path>", "Models directory (overrides PYCOMFY_MODELS_DIR)")
    .addHelpText("after", modelsFooter("edit", "video"))
    .action((opts) => {
      validateModel("edit", "video", opts.model);
      notImplemented("edit", "video", opts.model);
    });
}
