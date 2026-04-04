// Upscale command and its subcommands (image).
// Action handlers follow: validate → resolveModelsDir → buildArgs → spawnPipeline.

import { Command } from "commander";
import { readConfig } from "../config";
import { spawnPipeline, runAsync } from "../runner";
import { resolveModelsDir } from "../utils";
import { getModels, getScript } from "../models/registry";
import { buildUpscaleImageArgs, type UpscaleImageOpts } from "../models/image";
import { modelsFooter, validateModel, notImplemented } from "../utils";

export function registerUpscale(program: Command): void {
  const upscale = program
    .command("upscale")
    .description("Upscale existing media")
    .usage("<media> [options]");

  upscale
    .command("image")
    .description("Upscale an image")
    .requiredOption("--model <name>", `Model to use (choices: ${getModels("upscale", "image").join(", ")})`)
    .requiredOption("--prompt <text>", "Text prompt")
    .requiredOption("--input <path>", "Path to the input image file to upscale")
    .option("--checkpoint <file>", "Base checkpoint filename (overrides PYCOMFY_CHECKPOINT)")
    .option("--esrgan-checkpoint <file>", "ESRGAN checkpoint filename (required for esrgan; overrides PYCOMFY_ESRGAN_CHECKPOINT)")
    .option("--latent-upscale-checkpoint <file>", "Latent upscale checkpoint filename (required for latent_upscale; overrides PYCOMFY_LATENT_UPSCALE_CHECKPOINT)")
    .option("--negative-prompt <text>", "Negative prompt (what to avoid in the output)")
    .option("--width <pixels>", "Image width in pixels", "768")
    .option("--height <pixels>", "Image height in pixels", "768")
    .option("--steps <n>", "Number of sampling steps", "20")
    .option("--cfg <value>", "CFG guidance scale", "7")
    .option("--seed <n>", "Random seed for reproducibility")
    .option("--output <path>", "Output file path", "output.png")
    .option("--output-base <path>", "Intermediate base image before upscaling", "output_base.png")
    .option("--models-dir <path>", "Models directory (overrides PYCOMFY_MODELS_DIR)")
    .option("--async", "Queue job and return a job ID immediately (non-blocking)")
    .addHelpText("after", modelsFooter("upscale", "image"))
    .action(async (opts) => {
      validateModel("upscale", "image", opts.model);
      const script = getScript("upscale", "image", opts.model);
      if (!script) notImplemented("upscale", "image", opts.model);

      // Resolve --checkpoint from flag or env
      const checkpoint = opts.checkpoint ?? process.env["PYCOMFY_CHECKPOINT"];
      if (!checkpoint) {
        console.error("Error: --checkpoint is required (or set PYCOMFY_CHECKPOINT)");
        process.exit(1);
      }

      // Model-specific required flags
      if (opts.model === "esrgan") {
        const esrgan = opts.esrganCheckpoint ?? process.env["PYCOMFY_ESRGAN_CHECKPOINT"];
        if (!esrgan) {
          console.error("Error: --esrgan-checkpoint is required for esrgan (or set PYCOMFY_ESRGAN_CHECKPOINT)");
          process.exit(1);
        }
        opts.esrganCheckpoint = esrgan;
      }
      if (opts.model === "latent_upscale") {
        const latent = opts.latentUpscaleCheckpoint ?? process.env["PYCOMFY_LATENT_UPSCALE_CHECKPOINT"];
        if (!latent) {
          console.error("Error: --latent-upscale-checkpoint is required for latent_upscale (or set PYCOMFY_LATENT_UPSCALE_CHECKPOINT)");
          process.exit(1);
        }
        opts.latentUpscaleCheckpoint = latent;
      }

      const modelsDir = resolveModelsDir(opts.modelsDir);
      const upscaleOpts: UpscaleImageOpts = {
        model:                   opts.model,
        prompt:                  opts.prompt,
        input:                   opts.input,
        negativePrompt:          opts.negativePrompt,
        checkpoint,
        esrganCheckpoint:        opts.esrganCheckpoint,
        latentUpscaleCheckpoint: opts.latentUpscaleCheckpoint,
        width:                   opts.width,
        height:                  opts.height,
        steps:                   opts.steps,
        cfg:                     opts.cfg,
        seed:                    opts.seed,
        output:                  opts.output,
        outputBase:              opts.outputBase,
      };
      const args = buildUpscaleImageArgs(upscaleOpts, modelsDir);
      if (opts.async) {
        await runAsync("upscale", "image", opts.model, script, args, readConfig());
      } else {
        await spawnPipeline(script, args, readConfig());
      }
    });
}
