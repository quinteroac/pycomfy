// Create command and all its subcommands (image, video, audio).
// Action handlers follow: validate → resolveModelsDir → buildArgs → spawnPipeline.
// All model-specific special cases are encapsulated in the models/ builders.

import { existsSync } from "fs";
import { Command } from "commander";
import { readConfig } from "../config";
import { spawnPipeline } from "../runner";
import { resolveModelsDir } from "../utils";
import { getModels, getScript, getModelConfig, getModelDefaults } from "../models/registry";
import { buildArgs as buildImageArgs, type ImageOpts } from "../models/image";
import { buildArgs as buildVideoArgs, type VideoOpts } from "../models/video";
import { buildArgs as buildAudioArgs, type AudioOpts } from "../models/audio";

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
    .option("--width <pixels>", "Image width in pixels")
    .option("--height <pixels>", "Image height in pixels")
    .option("--steps <n>", "Number of sampling steps")
    .option("--cfg <value>", "CFG guidance scale")
    .option("--seed <n>", "Random seed for reproducibility")
    .option("--output <path>", "Output file path", "output.png")
    .option("--models-dir <path>", "Models directory (overrides PYCOMFY_MODELS_DIR)")
    .addHelpText("after", modelsFooter("create", "image"))
    .action(async (opts) => {
      validateModel("create", "image", opts.model);
      const script = getScript("create", "image", opts.model);
      if (!script) notImplemented("create", "image", opts.model);
      const modelsDir = resolveModelsDir(opts.modelsDir);
      const defaults = getModelDefaults("image", opts.model);
      const imageOpts: ImageOpts = {
        model:          opts.model,
        prompt:         opts.prompt,
        negativePrompt: opts.negativePrompt,
        width:          opts.width  ?? (defaults?.width  != null ? String(defaults.width)  : "1024"),
        height:         opts.height ?? (defaults?.height != null ? String(defaults.height) : "1024"),
        steps:          opts.steps  ?? (defaults?.steps  != null ? String(defaults.steps)  : "20"),
        cfg:            opts.cfg    ?? (defaults?.cfg    != null ? String(defaults.cfg)    : "7"),
        seed:           opts.seed,
        output:         opts.output,
      };
      const args = buildImageArgs(imageOpts, modelsDir);
      await spawnPipeline(script, args, readConfig());
    });

  create
    .command("video")
    .description("Generate a video from a text prompt")
    .requiredOption("--model <name>", `Model to use (choices: ${getModels("create", "video").join(", ")})`)
    .requiredOption("--prompt <text>", "Text prompt describing the video to generate")
    .option("--input <path>", "Input image path for image-to-video (ltx2, ltx23, wan21, wan22)")
    .option("--width <pixels>", "Video width in pixels")
    .option("--height <pixels>", "Video height in pixels")
    .option("--length <frames>", "Number of frames to generate")
    .option("--steps <n>", "Number of sampling steps")
    .option("--cfg <value>", "CFG guidance scale")
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
      const defaults = getModelDefaults("video", opts.model);
      const videoOpts: VideoOpts = {
        model:  opts.model,
        prompt: opts.prompt,
        input:  opts.input,
        width:  opts.width  ?? (defaults?.width  != null ? String(defaults.width)  : "832"),
        height: opts.height ?? (defaults?.height != null ? String(defaults.height) : "480"),
        length: opts.length ?? (defaults?.length != null ? String(defaults.length) : "81"),
        steps:  opts.steps  ?? (defaults?.steps  != null ? String(defaults.steps)  : "30"),
        cfg:    opts.cfg    ?? (defaults?.cfg    != null ? String(defaults.cfg)    : "6"),
        seed:   opts.seed,
        output: opts.output,
      };
      const args = buildVideoArgs(videoOpts, modelsDir);
      await spawnPipeline(script, args, readConfig());
    });

  create
    .command("audio")
    .description("Generate audio from a text prompt")
    .requiredOption("--model <name>", `Model to use (choices: ${getModels("create", "audio").join(", ")})`)
    .requiredOption("--prompt <text>", "Text prompt describing the audio to generate")
    .option("--length <seconds>", "Duration in seconds")
    .option("--steps <n>", "Number of sampling steps")
    .option("--cfg <value>", "CFG guidance scale")
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
      const defaults = getModelDefaults("audio", opts.model);
      const audioOpts: AudioOpts = {
        model:        opts.model,
        prompt:       opts.prompt,
        length:       opts.length ?? (defaults?.length != null ? String(defaults.length) : "30"),
        steps:        opts.steps  ?? (defaults?.steps  != null ? String(defaults.steps)  : "60"),
        cfg:          opts.cfg    ?? (defaults?.cfg    != null ? String(defaults.cfg)    : "2"),
        bpm:          opts.bpm,
        lyrics:       opts.lyrics,
        seed:         opts.seed,
        output:       opts.output,
        unet:         opts.unet,
        vae:          opts.vae,
        textEncoder1: opts.textEncoder1,
        textEncoder2: opts.textEncoder2,
      };
      const args = buildAudioArgs(audioOpts, modelsDir);
      await spawnPipeline(script, args, readConfig());
    });
}
