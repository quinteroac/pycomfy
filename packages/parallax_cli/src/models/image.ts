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
