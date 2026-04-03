import { existsSync, mkdirSync, readFileSync, writeFileSync } from "fs";
import { dirname, join } from "path";
import { homedir } from "os";

export interface ParallaxConfig {
  repoRoot?: string;
  modelsDir?: string;
  uvPath?: string;
  variant?: string;
  installedAt?: string;
}

const CONFIG_PATH = join(homedir(), ".config", "parallax", "config.json");

export function configExists(): boolean {
  return existsSync(CONFIG_PATH);
}

export function readConfig(): ParallaxConfig {
  let stored: ParallaxConfig = {};

  if (configExists()) {
    try {
      stored = JSON.parse(readFileSync(CONFIG_PATH, "utf-8")) as ParallaxConfig;
    } catch {
      // malformed JSON — treat as empty
    }
  }

  // Env vars take precedence over stored values (backward compat for CI)
  return {
    ...stored,
    ...(process.env.PARALLAX_REPO_ROOT !== undefined && {
      repoRoot: process.env.PARALLAX_REPO_ROOT,
    }),
    ...(process.env.PYCOMFY_MODELS_DIR !== undefined && {
      modelsDir: process.env.PYCOMFY_MODELS_DIR,
    }),
  };
}

export function writeConfig(config: ParallaxConfig): void {
  const dir = dirname(CONFIG_PATH);
  if (!existsSync(dir)) {
    mkdirSync(dir, { recursive: true });
  }
  writeFileSync(CONFIG_PATH, JSON.stringify(config, null, 2), "utf-8");
}
