import { join } from "path";
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
