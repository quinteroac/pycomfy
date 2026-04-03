// Create command and all its subcommands (image, video, audio).
// Action handlers follow: validate → resolveModelsDir → buildArgs → spawnPipeline.
// All model-specific special cases are encapsulated in the models/ builders.

import { existsSync } from "fs";
import { Command } from "commander";
import { readConfig } from "../config";
import { spawnPipeline } from "../runner";
import { resolveModelsDir } from "../utils";
import { getModels, getScript, getModelConfig } from "../models/registry";
import { buildArgs as buildImageArgs } from "../models/image";
import { buildArgs as buildVideoArgs } from "../models/video";
import { buildArgs as buildAudioArgs } from "../models/audio";

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

export function registerCreate(program: Command): void {
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
      if (!script) notImplemented("create", "image", opts.model);
      const modelsDir = resolveModelsDir(opts.modelsDir);
      const args = buildImageArgs(opts, modelsDir);
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
      const script = modelConfig ? (useI2v ? modelConfig.i2v! : modelConfig.t2v) : undefined;
      if (!script) notImplemented("create", "video", opts.model);
      const modelsDir = resolveModelsDir(opts.modelsDir);
      const args = buildVideoArgs(opts, modelsDir);
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
    .option("--unet <path>", "UNet model component path (overrides PYCOMFY_ACE_UNET)")
    .option("--vae <path>", "VAE model component path (overrides PYCOMFY_ACE_VAE)")
    .option("--text-encoder-1 <path>", "Text encoder 1 path (overrides PYCOMFY_ACE_TEXT_ENCODER_1)")
    .option("--text-encoder-2 <path>", "Text encoder 2 path (overrides PYCOMFY_ACE_TEXT_ENCODER_2)")
    .addHelpText("after", modelsFooter("create", "audio"))
    .action(async (opts) => {
      validateModel("create", "audio", opts.model);
      const script = getScript("create", "audio", opts.model);
      if (!script) notImplemented("create", "audio", opts.model);
      const modelsDir = resolveModelsDir(opts.modelsDir);
      const args = buildAudioArgs(opts, modelsDir);
      await spawnPipeline(script, args, readConfig());
    });
}
