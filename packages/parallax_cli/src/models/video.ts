// Argument builder for video generation pipelines.
// Handles omitSteps, per-model cfgFlag, and --image vs --input mapping so
// action handlers contain no inline model logic.

import { getModelConfig } from "./registry";

export interface VideoOpts {
  model: string;
  prompt: string;
  input?: string;
  width: string;
  height: string;
  length: string;
  steps: string;
  cfg: string;
  seed?: string;
  output: string;
}

// Build the CLI args array for a `create video` invocation.
//
// Key behaviours:
//  - When --input is provided and the model has an i2v script, the source
//    frame is forwarded as --image (the flag the Python i2v scripts expect).
//  - Distilled models (omitSteps: true) receive no --steps argument.
//  - Each model may declare a custom cfg flag (e.g. --cfg-pass1 for ltx2).
export function buildArgs(opts: VideoOpts, modelsDir: string): string[] {
  const modelConfig = getModelConfig("video", opts.model);
  const useI2v = opts.input !== undefined && modelConfig?.i2v !== undefined;

  const args: string[] = [
    "--models-dir", modelsDir,
    "--prompt", opts.prompt,
    "--width", opts.width,
    "--height", opts.height,
    "--length", opts.length,
    "--output", opts.output,
  ];

  // i2v scripts expect --image, not --input
  if (useI2v) args.push("--image", opts.input!);

  // Distilled models skip the steps argument entirely
  if (!modelConfig?.omitSteps) args.push("--steps", opts.steps);

  // Per-model cfg flag name (defaults to --cfg)
  args.push(modelConfig?.cfgFlag ?? "--cfg", opts.cfg);

  if (opts.seed !== undefined) args.push("--seed", opts.seed);

  return args;
}
