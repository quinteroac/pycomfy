#!/usr/bin/env bun
import { existsSync } from "fs";
import { Command } from "commander";
import { readConfig } from "./config";
import { spawnPipeline } from "./runner";
import {
  getModels,
  getScript,
  getModelConfig,
} from "./models/registry";

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

const create = program
  .command("create")
  .description("Generate media from a text prompt")
  .usage("<media> [options]");

create
  .command("image")
  .description("Generate an image from a text prompt")
  .requiredOption("--model <name>", `Model to use (choices: ${getModels("create", "image").join(", ")})`)
  .requiredOption("--prompt <text>", "Text prompt describing the image to generate")
  .option("--negative-prompt <text>", "Negative prompt (what to avoid in the output)")
  .option("--width <pixels>", "Image width in pixels", "1024")
  .option("--height <pixels>", "Image height in pixels", "1024")
  .option("--steps <n>", "Number of sampling steps", "20")
  .option("--cfg <value>", "CFG guidance scale", "7")
  .option("--seed <n>", "Random seed for reproducibility")
  .option("--output <path>", "Output file path", "output.png")
  .option("--models-dir <path>", "Models directory (overrides PYCOMFY_MODELS_DIR)")
  .addHelpText("after", modelsFooter("create", "image"))
  .action(async (opts) => {
    validateModel("create", "image", opts.model);

    const script = getScript("create", "image", opts.model);
    if (!script) {
      notImplemented("create", "image", opts.model);
    }

    const modelsDir = opts.modelsDir ?? process.env.PYCOMFY_MODELS_DIR;
    if (!modelsDir) {
      console.error("Error: --models-dir or PYCOMFY_MODELS_DIR is required");
      process.exit(1);
    }

    const args: string[] = [
      "--models-dir", modelsDir,
      "--prompt", opts.prompt,
      "--width", opts.width,
      "--height", opts.height,
      "--steps", opts.steps,
      "--output", opts.output,
    ];

    // z_image turbo.py accepts neither --negative-prompt nor --cfg; omit both to
    // avoid "unrecognized arguments" errors from the Python script.
    if (opts.model !== "z_image") {
      if (opts.negativePrompt) args.push("--negative-prompt", opts.negativePrompt);
      args.push("--cfg", opts.cfg);
    }

    if (opts.seed !== undefined) args.push("--seed", opts.seed);

    await spawnPipeline(script, args, readConfig());
  });

create
  .command("video")
  .description("Generate a video from a text prompt")
  .requiredOption("--model <name>", `Model to use (choices: ${getModels("create", "video").join(", ")})`)
  .requiredOption("--prompt <text>", "Text prompt describing the video to generate")
  .option("--input <path>", "Input image path for image-to-video (ltx2, ltx23, wan21, wan22)")
  .option("--width <pixels>", "Video width in pixels", "832")
  .option("--height <pixels>", "Video height in pixels", "480")
  .option("--length <frames>", "Number of frames to generate", "81")
  .option("--steps <n>", "Number of sampling steps", "30")
  .option("--cfg <value>", "CFG guidance scale", "6")
  .option("--seed <n>", "Random seed for reproducibility")
  .option("--output <path>", "Output file path", "output.mp4")
  .option("--models-dir <path>", "Models directory (overrides PYCOMFY_MODELS_DIR)")
  .addHelpText("after", modelsFooter("create", "video"))
  .action(async (opts) => {
    validateModel("create", "video", opts.model);

    if (opts.input !== undefined && !existsSync(opts.input)) {
      console.error(`Error: input file not found: ${opts.input}`);
      process.exit(1);
    }

    const modelConfig = getModelConfig("video", opts.model);
    const useI2v = opts.input !== undefined && modelConfig?.i2v !== undefined;
    const script = modelConfig
      ? (useI2v ? modelConfig.i2v! : modelConfig.t2v)
      : undefined;

    if (!script) {
      notImplemented("create", "video", opts.model);
    }

    const modelsDir = opts.modelsDir ?? process.env.PYCOMFY_MODELS_DIR;
    if (!modelsDir) {
      console.error("Error: --models-dir or PYCOMFY_MODELS_DIR is required");
      process.exit(1);
    }

    const args: string[] = [
      "--models-dir", modelsDir,
      "--prompt", opts.prompt,
      "--width", opts.width,
      "--height", opts.height,
      "--length", opts.length,
      "--output", opts.output,
    ];

    // i2v.py expects --image (not --input) for the source frame
    if (useI2v) args.push("--image", opts.input!);

    // Distilled models omit --steps; all others receive it
    if (!modelConfig?.omitSteps) args.push("--steps", opts.steps);

    // Per-model cfg flag name (e.g. --cfg-pass1 for ltx2)
    args.push(modelConfig?.cfgFlag ?? "--cfg", opts.cfg);

    if (opts.seed !== undefined) args.push("--seed", opts.seed);

    await spawnPipeline(script, args, readConfig());
  });

create
  .command("audio")
  .description("Generate audio from a text prompt")
  .requiredOption("--model <name>", `Model to use (choices: ${getModels("create", "audio").join(", ")})`)
  .requiredOption("--prompt <text>", "Text prompt describing the audio to generate")
  .option("--length <seconds>", "Duration in seconds", "30")
  .option("--steps <n>", "Number of sampling steps", "60")
  .option("--cfg <value>", "CFG guidance scale", "2")
  .option("--bpm <n>", "Beats per minute", "120")
  .option("--lyrics <text>", "Lyrics text (ace_step)", "")
  .option("--seed <n>", "Random seed for reproducibility")
  .option("--output <path>", "Output file path", "output.wav")
  .option("--models-dir <path>", "Models directory (overrides PYCOMFY_MODELS_DIR)")
  .addHelpText("after", modelsFooter("create", "audio"))
  .action(async (opts) => {
    validateModel("create", "audio", opts.model);

    const script = getScript("create", "audio", opts.model);
    if (!script) {
      notImplemented("create", "audio", opts.model);
    }

    const modelsDir = opts.modelsDir ?? process.env.PYCOMFY_MODELS_DIR;
    if (!modelsDir) {
      console.error("Error: --models-dir or PYCOMFY_MODELS_DIR is required");
      process.exit(1);
    }

    const args: string[] = [
      "--models-dir", modelsDir,
      "--tags", opts.prompt,         // CLI --prompt → script --tags
      "--duration", opts.length,     // CLI --length → script --duration
      "--steps", opts.steps,
      "--cfg", opts.cfg,
      "--bpm", opts.bpm,
      "--lyrics", opts.lyrics,
      "--output", opts.output,
    ];

    if (opts.seed !== undefined) args.push("--seed", opts.seed);

    await spawnPipeline(script, args, readConfig());
  });

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
