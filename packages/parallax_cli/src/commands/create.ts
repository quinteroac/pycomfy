// Create command and all its subcommands (image, video, audio).
// Action handlers follow: validate → resolveModelsDir → buildArgs → spawnPipeline.
// All model-specific special cases are encapsulated in the models/ builders.

import { existsSync } from "fs";
import { Command } from "commander";
import { readConfig } from "../config";
import { spawnPipeline, runAsync } from "../runner";
import { resolveModelsDir } from "../utils";
import { getModels, getScript, getModelConfig, getModelDefaults, type ModelDefaults } from "../models/registry";
import { buildArgs as buildImageArgs, type ImageOpts } from "../models/image";
import { buildArgs as buildVideoArgs, type VideoOpts } from "../models/video";
import { buildArgs as buildAudioArgs, type AudioOpts } from "../models/audio";
import { modelsFooter, validateModel, notImplemented, resolveParam } from "../utils";

// Columns per media type: [key in ModelDefaults, display label]
const DEFAULTS_COLUMNS: Record<string, Array<[keyof ModelDefaults, string]>> = {
  video: [["width","width"],["height","height"],["length","length"],["fps","fps"],["steps","steps"],["cfg","cfg"]],
  image: [["width","width"],["height","height"],["steps","steps"],["cfg","cfg"]],
  audio: [["length","duration"],["steps","steps"],["cfg","cfg"],["bpm","bpm"]],
};

export function buildDefaultsTable(media: string): string {
  const cols = DEFAULTS_COLUMNS[media];
  if (!cols) return "";
  const models = getModels("create", media).filter(m => getModelDefaults(media, m) !== undefined);
  if (models.length === 0) return "";

  const headers = ["model", ...cols.map(([, label]) => label)];
  const rows: string[][] = models.map(model => {
    const d = getModelDefaults(media, model)!;
    return [model, ...cols.map(([key]) => d[key] != null ? String(d[key]) : "—")];
  });

  const widths = headers.map((h, i) => Math.max(h.length, ...rows.map(r => r[i].length)));
  const pad = (s: string, w: number) => s.padEnd(w);
  const headerRow = "  " + headers.map((h, i) => pad(h, widths[i])).join("  ");
  const sepRow = "  " + "─".repeat(headerRow.length - 2);
  const dataRows = rows.map(row => "  " + row.map((cell, i) => pad(cell, widths[i])).join("  "));

  return "\nDefaults per model:\n\n" + headerRow + "\n" + sepRow + "\n" + dataRows.join("\n");
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
    .option("--async", "Queue job and return a job ID immediately (non-blocking)")
    .addHelpText("after", modelsFooter("create", "image"))
    .addHelpText("after", buildDefaultsTable("image"))
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
        width:          resolveParam(opts.width,  defaults?.width,  "width"),
        height:         resolveParam(opts.height, defaults?.height, "height"),
        steps:          resolveParam(opts.steps,  defaults?.steps,  "steps"),
        cfg:            resolveParam(opts.cfg,    defaults?.cfg,    "cfg"),
        seed:           opts.seed,
        output:         opts.output,
      };
      const args = buildImageArgs(imageOpts, modelsDir);
      if (opts.async) {
        await runAsync("create", "image", opts.model, script, args, readConfig());
      } else {
        await spawnPipeline(script, args, readConfig());
      }
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
    .option("--async", "Queue job and return a job ID immediately (non-blocking)")
    .addHelpText("after", modelsFooter("create", "video"))
    .addHelpText("after", buildDefaultsTable("video"))
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
        width:  resolveParam(opts.width,  defaults?.width,  "width"),
        height: resolveParam(opts.height, defaults?.height, "height"),
        length: resolveParam(opts.length, defaults?.length, "length"),
        steps:  resolveParam(opts.steps,  defaults?.steps,  "steps"),
        cfg:    resolveParam(opts.cfg,    defaults?.cfg,    "cfg"),
        seed:   opts.seed,
        output: opts.output,
      };
      const args = buildVideoArgs(videoOpts, modelsDir);
      if (opts.async) {
        await runAsync("create", "video", opts.model, script, args, readConfig());
      } else {
        await spawnPipeline(script, args, readConfig());
      }
    });

  create
    .command("audio")
    .description("Generate audio from a text prompt")
    .requiredOption("--model <name>", `Model to use (choices: ${getModels("create", "audio").join(", ")})`)
    .requiredOption("--prompt <text>", "Text prompt describing the audio to generate")
    .option("--length <seconds>", "Duration in seconds")
    .option("--steps <n>", "Number of sampling steps")
    .option("--cfg <value>", "CFG guidance scale")
    .option("--bpm <n>", "Beats per minute")
    .option("--lyrics <text>", "Lyrics text (ace_step)", "")
    .option("--seed <n>", "Random seed for reproducibility")
    .option("--output <path>", "Output file path", "output.wav")
    .option("--models-dir <path>", "Models directory (overrides PYCOMFY_MODELS_DIR)")
    .option("--unet <path>", "UNet model component path (overrides PYCOMFY_ACE_UNET)")
    .option("--vae <path>", "VAE model component path (overrides PYCOMFY_ACE_VAE)")
    .option("--text-encoder-1 <path>", "Text encoder 1 path (overrides PYCOMFY_ACE_TEXT_ENCODER_1)")
    .option("--text-encoder-2 <path>", "Text encoder 2 path (overrides PYCOMFY_ACE_TEXT_ENCODER_2)")
    .option("--async", "Queue job and return a job ID immediately (non-blocking)")
    .addHelpText("after", modelsFooter("create", "audio"))
    .addHelpText("after", buildDefaultsTable("audio"))
    .action(async (opts) => {
      validateModel("create", "audio", opts.model);
      const script = getScript("create", "audio", opts.model);
      if (!script) notImplemented("create", "audio", opts.model);
      const modelsDir = resolveModelsDir(opts.modelsDir);
      const defaults = getModelDefaults("audio", opts.model);
      const audioOpts: AudioOpts = {
        model:        opts.model,
        prompt:       opts.prompt,
        length:       resolveParam(opts.length, defaults?.length, "length"),
        steps:        resolveParam(opts.steps,  defaults?.steps,  "steps"),
        cfg:          resolveParam(opts.cfg,    defaults?.cfg,    "cfg"),
        bpm:          resolveParam(opts.bpm,    defaults?.bpm,    "bpm"),
        lyrics:       opts.lyrics,
        seed:         opts.seed,
        output:       opts.output,
        unet:         opts.unet,
        vae:          opts.vae,
        textEncoder1: opts.textEncoder1,
        textEncoder2: opts.textEncoder2,
      };
      const args = buildAudioArgs(audioOpts, modelsDir);
      if (opts.async) {
        await runAsync("create", "audio", opts.model, script, args, readConfig());
      } else {
        await spawnPipeline(script, args, readConfig());
      }
    });
}
