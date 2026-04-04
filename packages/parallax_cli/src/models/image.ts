// Argument builder for image generation pipelines.
// All image-specific special cases are encapsulated here so action
// handlers contain no inline model logic.

export interface ImageOpts {
  model: string;
  prompt: string;
  negativePrompt?: string;
  width: string;
  height: string;
  steps: string;
  cfg: string;
  seed?: string;
  output: string;
}

// Build the CLI args array for a `create image` invocation.
// z_image turbo.py accepts neither --negative-prompt nor --cfg; both are
// omitted when model === "z_image" to avoid "unrecognized arguments" errors.
export function buildArgs(opts: ImageOpts, modelsDir: string): string[] {
  const args: string[] = [
    "--models-dir", modelsDir,
    "--prompt", opts.prompt,
    "--width", opts.width,
    "--height", opts.height,
    "--steps", opts.steps,
    "--output", opts.output,
  ];

  if (opts.model !== "z_image") {
    if (opts.negativePrompt) args.push("--negative-prompt", opts.negativePrompt);
    args.push("--cfg", opts.cfg);
  }

  if (opts.seed !== undefined) args.push("--seed", opts.seed);

  return args;
}

export interface EditImageOpts {
  model: string;
  prompt: string;
  input: string;
  width: string;
  height: string;
  steps?: string;   // qwen: omit to let Python auto-detect (4 w/ LoRA, 40 w/o)
  cfg?: string;     // qwen: omit to let Python auto-detect (1.0 w/ LoRA, 4.0 w/o)
  seed?: string;
  output: string;
  subjectImage?: string; // flux_9b_kv only
  image2?: string;       // qwen only
  image3?: string;       // qwen only
  noLora?: boolean;      // qwen only
}

export interface UpscaleImageOpts {
  model: string;
  prompt: string;
  input: string;
  negativePrompt?: string;
  checkpoint: string;
  esrganCheckpoint?: string;      // esrgan only
  latentUpscaleCheckpoint?: string; // latent_upscale only
  width: string;
  height: string;
  steps: string;
  cfg: string;
  seed?: string;
  output: string;
  outputBase: string;
}

// Build the CLI args array for an `upscale image` invocation.
//
// Both esrgan and latent_upscale share most flags:
//  --models-dir, --checkpoint, --prompt, --negative-prompt, --width, --height,
//  --steps, --cfg, --seed, --output, --output-base
// Model-specific:
//  - esrgan: also passes --esrgan-checkpoint
//  - latent_upscale: also passes --latent-upscale-checkpoint
export function buildUpscaleImageArgs(opts: UpscaleImageOpts, modelsDir: string): string[] {
  const args: string[] = [
    "--models-dir", modelsDir,
    "--checkpoint", opts.checkpoint,
    "--prompt", opts.prompt,
    "--input", opts.input,
    "--width", opts.width,
    "--height", opts.height,
    "--steps", opts.steps,
    "--cfg", opts.cfg,
    "--output", opts.output,
    "--output-base", opts.outputBase,
  ];

  if (opts.negativePrompt !== undefined) args.push("--negative-prompt", opts.negativePrompt);
  if (opts.seed !== undefined) args.push("--seed", opts.seed);

  if (opts.model === "esrgan" && opts.esrganCheckpoint !== undefined) {
    args.push("--esrgan-checkpoint", opts.esrganCheckpoint);
  }
  if (opts.model === "latent_upscale" && opts.latentUpscaleCheckpoint !== undefined) {
    args.push("--latent-upscale-checkpoint", opts.latentUpscaleCheckpoint);
  }

  return args;
}

// Build the CLI args array for an `edit image` invocation.
//
// Model-specific behaviours:
//  - flux variants (not flux_9b_kv): --image <input>, no --cfg forwarded.
//  - flux_9b_kv: same as other flux variants, plus --subject-image.
//  - qwen: --image <input>, --cfg forwarded, --output becomes --output-prefix
//    (with trailing .png stripped), optional --image2/--image3/--no-lora.
export function buildEditImageArgs(opts: EditImageOpts, modelsDir: string): string[] {
  const args: string[] = ["--models-dir", modelsDir];

  if (opts.model === "qwen") {
    args.push("--image", opts.input);
    args.push("--prompt", opts.prompt);
    if (opts.steps !== undefined) args.push("--steps", opts.steps);
    if (opts.cfg !== undefined) args.push("--cfg", opts.cfg);
    if (opts.seed !== undefined) args.push("--seed", opts.seed);
    const prefix = opts.output.endsWith(".png") ? opts.output.slice(0, -4) : opts.output;
    args.push("--output-prefix", prefix);
    if (opts.image2 !== undefined) args.push("--image2", opts.image2);
    if (opts.image3 !== undefined) args.push("--image3", opts.image3);
    if (opts.noLora) args.push("--no-lora");
  } else {
    // All flux variants
    args.push("--prompt", opts.prompt);
    args.push("--image", opts.input);
    args.push("--width", opts.width);
    args.push("--height", opts.height);
    args.push("--steps", opts.steps);
    if (opts.seed !== undefined) args.push("--seed", opts.seed);
    args.push("--output", opts.output);
    if (opts.model === "flux_9b_kv" && opts.subjectImage !== undefined) {
      args.push("--subject-image", opts.subjectImage);
    }
  }

  return args;
}
