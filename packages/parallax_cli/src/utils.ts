import { readConfig } from "./config";

/** Reformat commander's missing-required-option errors to "Error: --flag is required" */
export function formatRequiredFlagError(msg: string): string {
  return msg.replace(
    /error: required option '(--[a-z-]+)[^']*' not specified/,
    "Error: $1 is required"
  );
}

// Resolves --models-dir with priority: flag > stored config > env var.
export function resolveModelsDir(flag?: string): string {
  const modelsDir = flag ?? readConfig().modelsDir;
  if (!modelsDir) {
    console.error("Error: --models-dir or PYCOMFY_MODELS_DIR is required");
    process.exit(1);
  }
  return modelsDir;
}
