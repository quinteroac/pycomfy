#!/usr/bin/env bun
import { existsSync } from "fs";
import { Command } from "commander";
import { readConfig } from "./config";
import { spawnPipeline } from "./runner";

// Known models per action+media — single source of truth for help text and validation
const MODELS: Record<string, string[]> = {
  "create image": ["sdxl", "anima", "z_image", "flux_klein", "qwen"],
  "create video": ["ltx2", "ltx23", "wan21", "wan22"],
  "create audio": ["ace_step"],
  "edit image": ["qwen"],
  "edit video": ["wan21", "wan22"],
};

// Script paths (relative to PARALLAX_REPO_ROOT) for each implemented image model.
// Models absent from this map fall through to notImplemented().
const IMAGE_SCRIPTS: Partial<Record<string, string>> = {
  sdxl:    "examples/image/generation/sdxl/t2i.py",
  anima:   "examples/image/generation/anima/t2i.py",
  z_image: "examples/image/generation/z_image/turbo.py",
};

// Per-model video config: script paths and flag-forwarding rules.
interface VideoModelConfig {
  t2v: string;
  i2v?: string;       // i2v.py path; defined for models that support image-to-video
  cfgFlag: string;    // "--cfg" or "--cfg-pass1" depending on model interface
  omitSteps?: true;   // set for distilled models that accept no --steps argument
}

const VIDEO_MODEL_CONFIG: Record<string, VideoModelConfig> = {
  ltx2:  { t2v: "examples/video/ltx/ltx2/t2v.py",  i2v: "examples/video/ltx/ltx2/i2v.py",  cfgFlag: "--cfg-pass1" },
  ltx23: { t2v: "examples/video/ltx/ltx23/t2v.py", i2v: "examples/video/ltx/ltx23/i2v.py", cfgFlag: "--cfg", omitSteps: true },
  wan21: { t2v: "examples/video/wan/wan21/t2v.py",  i2v: "examples/video/wan/wan21/i2v.py",  cfgFlag: "--cfg" },
  wan22: { t2v: "examples/video/wan/wan22/t2v.py",  i2v: "examples/video/wan/wan22/i2v.py",  cfgFlag: "--cfg" },
};

// Script paths for each implemented audio model.
const AUDIO_SCRIPTS: Partial<Record<string, string>> = {
  ace_step: "examples/audio/ace/t2a.py",
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
  .requiredOption("--model <name>", `Model to use (choices: ${MODELS["create image"].join(", ")})`)
  .requiredOption("--prompt <text>", "Text prompt describing the image to generate")
  .option("--negative-prompt <text>", "Negative prompt (what to avoid in the output)")
  .option("--width <pixels>", "Image width in pixels", "1024")
  .option("--height <pixels>", "Image height in pixels", "1024")
  .option("--steps <n>", "Number of sampling steps", "20")
  .option("--cfg <value>", "CFG guidance scale", "7")
  .option("--seed <n>", "Random seed for reproducibility")
  .option("--output <path>", "Output file path", "output.png")
  .option("--models-dir <path>", "Models directory (overrides PYCOMFY_MODELS_DIR)")
  .addHelpText("after", modelsFooter("create image"))
  .action(async (opts) => {
    validateModel("create image", opts.model);

    const script = IMAGE_SCRIPTS[opts.model];
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
  .requiredOption("--model <name>", `Model to use (choices: ${MODELS["create video"].join(", ")})`)
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
  .addHelpText("after", modelsFooter("create video"))
  .action(async (opts) => {
    validateModel("create video", opts.model);

    if (opts.input !== undefined && !existsSync(opts.input)) {
      console.error(`Error: input file not found: ${opts.input}`);
      process.exit(1);
    }

    const modelConfig = VIDEO_MODEL_CONFIG[opts.model];
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
  .requiredOption("--model <name>", `Model to use (choices: ${MODELS["create audio"].join(", ")})`)
  .requiredOption("--prompt <text>", "Text prompt describing the audio to generate")
  .option("--length <seconds>", "Duration in seconds", "30")
  .option("--steps <n>", "Number of sampling steps", "60")
  .option("--cfg <value>", "CFG guidance scale", "2")
  .option("--bpm <n>", "Beats per minute", "120")
  .option("--lyrics <text>", "Lyrics text (ace_step)", "")
  .option("--seed <n>", "Random seed for reproducibility")
  .option("--output <path>", "Output file path", "output.wav")
  .option("--models-dir <path>", "Models directory (overrides PYCOMFY_MODELS_DIR)")
  .addHelpText("after", modelsFooter("create audio"))
  .action(async (opts) => {
    validateModel("create audio", opts.model);

    const script = AUDIO_SCRIPTS[opts.model];
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
  .requiredOption("--model <name>", `Model to use (choices: ${MODELS["edit image"].join(", ")})`)
  .requiredOption("--prompt <text>", "Text prompt describing the desired edits")
  .requiredOption("--input <path>", "Path to the input image file")
  .option("--steps <n>", "Number of sampling steps", "20")
  .option("--cfg <value>", "CFG guidance scale", "7")
  .option("--seed <n>", "Random seed for reproducibility")
  .option("--output <path>", "Output file path", "output.png")
  .addHelpText("after", modelsFooter("edit image"))
  .action((opts) => { validateModel("edit image", opts.model); notImplemented("edit", "image", opts.model); });

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
  .action((opts) => { validateModel("edit video", opts.model); notImplemented("edit", "video", opts.model); });

if (process.argv.length <= 2) {
  program.help();
}

program.parse();
