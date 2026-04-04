import { readConfig } from "./config";
import { getModels } from "./models/registry";

// ---------------------------------------------------------------------------
// Shared CLI helpers — used across create, edit, and upscale commands.
// ---------------------------------------------------------------------------

export function modelsFooter(action: string, media: string): string {
  return `\nAvailable models: ${getModels(action, media).join(", ")}`;
}

export function validateModel(action: string, media: string, model: string): void {
  const known = getModels(action, media);
  if (!known.includes(model)) {
    console.error(
      `Error: unknown model "${model}" for ${action} ${media}. Known models: ${known.join(", ")}`
    );
    process.exit(1);
  }
}

export function notImplemented(action: string, media: string, model: string): never {
  console.log(`[parallax] ${action} ${media} --model ${model} — not yet implemented (coming soon)`);
  process.exit(0);
}

/**
 * Resolve a numeric registry default to a string, falling back to an explicit
 * runtime error when neither a user-supplied value nor a registry entry exists.
 * Removes the need for silent hardcoded last-resort strings in command handlers.
 */
export function resolveParam(
  userValue: string | undefined,
  registryValue: number | undefined,
  name: string,
): string {
  if (userValue !== undefined) return userValue;
  if (registryValue != null) return String(registryValue);
  console.error(`Internal error: model has no registry default for '${name}'`);
  process.exit(1);
}

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
