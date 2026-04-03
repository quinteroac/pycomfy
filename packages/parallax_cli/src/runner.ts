import { join } from "path";
import type { ParallaxConfig } from "./config";

// Spawn `uv run python <script>` with inherited stdio; exit with the child's exit code.
// repoRoot and uvPath are sourced from the resolved config object — no direct env access.
export async function spawnPipeline(
  scriptRelPath: string,
  args: string[],
  config: ParallaxConfig,
): Promise<void> {
  const { repoRoot, uvPath = "uv" } = config;

  if (!repoRoot) {
    console.error("Error: PARALLAX_REPO_ROOT is required");
    process.exit(1);
  }

  const proc = Bun.spawn(
    [uvPath, "run", "python", join(repoRoot, scriptRelPath), ...args],
    { stdin: "inherit", stdout: "inherit", stderr: "inherit", cwd: repoRoot },
  );

  process.exit(await proc.exited);
}
