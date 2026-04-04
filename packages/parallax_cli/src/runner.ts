import { join } from "path";
import { submitJob } from "@parallax/sdk/submit";
import type { ParallaxJobData } from "@parallax/sdk";
import type { ParallaxConfig } from "./config";

// Spawn `uv run python <script>` with inherited stdio; exit with the child's exit code.
// Script path is resolved from runtimeDir (installed mode) or repoRoot (dev mode).
export async function spawnPipeline(
  scriptRelPath: string,
  args: string[],
  config: ParallaxConfig,
): Promise<void> {
  const { runtimeDir, repoRoot, uvPath = "uv" } = config;

  const scriptBase = runtimeDir ?? repoRoot;

  if (!scriptBase) {
    console.error(
      "Error: no script directory configured — run `parallax install` to set runtimeDir, or set PARALLAX_REPO_ROOT",
    );
    process.exit(1);
  }

  const proc = Bun.spawn(
    [uvPath, "run", "python", join(scriptBase, scriptRelPath), ...args],
    { stdin: "inherit", stdout: "inherit", stderr: "inherit", cwd: scriptBase },
  );

  process.exit(await proc.exited);
}

// Pure helper — builds the job data payload for submitJob().
export function buildJobData(
  action: string,
  media: string,
  model: string,
  scriptRelPath: string,
  args: string[],
  scriptBase: string,
  uvPath: string,
): ParallaxJobData {
  return { action, media, model, script: scriptRelPath, args, scriptBase, uvPath };
}

// Pure helper — formats the async confirmation line printed to stdout.
export function formatAsyncMessage(jobId: string): string {
  return `Job ${jobId} queued\n  → parallax jobs watch ${jobId}`;
}

// Queue the pipeline as a background job and print the job ID, then exit 0.
export async function runAsync(
  action: string,
  media: string,
  model: string,
  scriptRelPath: string,
  args: string[],
  config: ParallaxConfig,
): Promise<void> {
  const { runtimeDir, repoRoot, uvPath = "uv" } = config;
  const scriptBase = runtimeDir ?? repoRoot;

  if (!scriptBase) {
    console.error(
      "Error: no script directory configured — run `parallax install` to set runtimeDir, or set PARALLAX_REPO_ROOT",
    );
    process.exit(1);
  }

  const jobId = await submitJob(
    buildJobData(action, media, model, scriptRelPath, args, scriptBase, uvPath),
  );

  console.log(formatAsyncMessage(jobId));
  process.exit(0);
}
