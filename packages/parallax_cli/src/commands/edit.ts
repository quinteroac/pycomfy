// Edit command and all its subcommands (image, video).
// Action handlers follow: validate → resolveModelsDir → buildArgs → spawnPipeline.

import { existsSync } from "fs";
import { Command } from "commander";
import { readConfig } from "../config";
import { spawnPipeline } from "../runner";
import { resolveModelsDir } from "../utils";
import { getModels, getScript } from "../models/registry";
import { buildEditImageArgs, type EditImageOpts } from "../models/image";
import { modelsFooter, validateModel, notImplemented } from "../utils";

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
    .option("--subject-image <path>", "Subject reference image (flux_9b_kv only)")
    .option("--width <pixels>", "Image width in pixels", "1024")
    .option("--height <pixels>", "Image height in pixels", "1024")
    .option("--steps <n>", "Number of sampling steps (qwen: auto 4 with LoRA, 40 without)")
    .option("--cfg <value>", "CFG guidance scale (qwen: auto 1.0 with LoRA, 4.0 without)")
    .option("--seed <n>", "Random seed for reproducibility")
    .option("--output <path>", "Output file path", "output.png")
    .option("--image2 <path>", "Second input image (qwen only)")
    .option("--image3 <path>", "Third input image (qwen only)")
    .option("--no-lora", "Disable LoRA (qwen only)")
    .option("--models-dir <path>", "Models directory (overrides PYCOMFY_MODELS_DIR)")
    .addHelpText("after", modelsFooter("edit", "image"))
    .action(async (opts) => {
      validateModel("edit", "image", opts.model);
      const script = getScript("edit", "image", opts.model);
      if (!script) notImplemented("edit", "image", opts.model);
      if (!existsSync(opts.input)) {
        console.error(`Error: input file not found: ${opts.input}`);
        process.exit(1);
      }
      if (opts.model === "flux_9b_kv") {
        if (!opts.subjectImage) {
          console.error("Error: --subject-image is required for flux_9b_kv");
          process.exit(1);
        }
        if (!existsSync(opts.subjectImage)) {
          console.error(`Error: subject-image file not found: ${opts.subjectImage}`);
          process.exit(1);
        }
      }
      const modelsDir = resolveModelsDir(opts.modelsDir);
      const editOpts: EditImageOpts = {
        model:        opts.model,
        prompt:       opts.prompt,
        input:        opts.input,
        width:        opts.width,
        height:       opts.height,
        steps:        opts.steps,
        cfg:          opts.cfg,
        seed:         opts.seed,
        output:       opts.output,
        subjectImage: opts.subjectImage,
        image2:       opts.image2,
        image3:       opts.image3,
        noLora:       !!opts.noLora,
      };
      const args = buildEditImageArgs(editOpts, modelsDir);
      await spawnPipeline(script, args, readConfig());
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
