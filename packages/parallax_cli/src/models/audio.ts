// Argument builder for audio generation pipelines.
// Handles the --prompt → --tags and --length → --duration flag remapping so
// action handlers contain no inline model logic.

export interface AudioOpts {
  model: string;
  prompt: string;
  length: string;
  steps: string;
  cfg: string;
  bpm: string;
  lyrics: string;
  seed?: string;
  output: string;
  unet?: string;
  vae?: string;
  textEncoder1?: string;
  textEncoder2?: string;
}

// Build the CLI args array for a `create audio` invocation.
//
// Flag remapping applied here:
//  - CLI --prompt is forwarded as --tags (ace_step script convention)
//  - CLI --length is forwarded as --duration (seconds, not frames)
// Component flags fall back to env vars when CLI flags are omitted.
export function buildArgs(opts: AudioOpts, modelsDir: string): string[] {
  const args: string[] = [
    "--models-dir", modelsDir,
    "--tags", opts.prompt,
    "--duration", opts.length,
    "--steps", opts.steps,
    "--cfg", opts.cfg,
    "--bpm", opts.bpm,
    "--lyrics", opts.lyrics,
    "--output", opts.output,
  ];

  if (opts.seed !== undefined) args.push("--seed", opts.seed);

  const unet = opts.unet ?? process.env.PYCOMFY_ACE_UNET;
  if (unet) args.push("--unet", unet);

  const vae = opts.vae ?? process.env.PYCOMFY_ACE_VAE;
  if (vae) args.push("--vae", vae);

  const te1 = opts.textEncoder1 ?? process.env.PYCOMFY_ACE_TEXT_ENCODER_1;
  if (te1) args.push("--text-encoder-1", te1);

  const te2 = opts.textEncoder2 ?? process.env.PYCOMFY_ACE_TEXT_ENCODER_2;
  if (te2) args.push("--text-encoder-2", te2);

  return args;
}
