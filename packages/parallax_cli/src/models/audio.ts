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
}

// Build the CLI args array for a `create audio` invocation.
//
// Flag remapping applied here:
//  - CLI --prompt is forwarded as --tags (ace_step script convention)
//  - CLI --length is forwarded as --duration (seconds, not frames)
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

  return args;
}
