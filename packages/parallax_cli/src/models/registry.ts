// Centralized model registry — single source of truth for all model metadata.
// Adding a new model requires editing only this file.

export const MODELS: Record<string, string[]> = {
  "create image": ["sdxl", "anima", "z_image", "flux_klein", "qwen"],
  "create video": ["ltx2", "ltx23", "wan21", "wan22"],
  "create audio": ["ace_step"],
  "edit image":   ["qwen"],
  "edit video":   ["wan21", "wan22"],
};

export const IMAGE_SCRIPTS: Partial<Record<string, string>> = {
  sdxl:    "examples/image/generation/sdxl/t2i.py",
  anima:   "examples/image/generation/anima/t2i.py",
  z_image: "examples/image/generation/z_image/turbo.py",
};

export interface ModelConfig {
  t2v: string;
  i2v?: string;
  cfgFlag: string;
  omitSteps?: true;
}

export const VIDEO_MODEL_CONFIG: Record<string, ModelConfig> = {
  ltx2:  { t2v: "examples/video/ltx/ltx2/t2v.py",  i2v: "examples/video/ltx/ltx2/i2v.py",  cfgFlag: "--cfg-pass1" },
  ltx23: { t2v: "examples/video/ltx/ltx23/t2v.py", i2v: "examples/video/ltx/ltx23/i2v.py", cfgFlag: "--cfg", omitSteps: true },
  wan21: { t2v: "examples/video/wan/wan21/t2v.py",  i2v: "examples/video/wan/wan21/i2v.py",  cfgFlag: "--cfg" },
  wan22: { t2v: "examples/video/wan/wan22/t2v.py",  i2v: "examples/video/wan/wan22/i2v.py",  cfgFlag: "--cfg" },
};

export const AUDIO_SCRIPTS: Partial<Record<string, string>> = {
  ace_step: "examples/audio/ace/t2a.py",
};

export function getModels(action: string, media: string): string[] {
  return MODELS[`${action} ${media}`] ?? [];
}

export function getScript(action: string, media: string, model: string): string | undefined {
  if (media === "image") return IMAGE_SCRIPTS[model];
  if (media === "audio") return AUDIO_SCRIPTS[model];
  if (media === "video") {
    const cfg = VIDEO_MODEL_CONFIG[model];
    if (!cfg) return undefined;
    // Default to t2v; callers decide i2v by inspecting getModelConfig
    return cfg.t2v;
  }
  return undefined;
}

export function getModelConfig(media: string, model: string): ModelConfig | undefined {
  if (media === "video") return VIDEO_MODEL_CONFIG[model];
  return undefined;
}
