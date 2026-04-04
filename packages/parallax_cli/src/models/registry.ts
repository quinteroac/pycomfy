// Centralized model registry — single source of truth for all model metadata.
// Adding a new model requires editing only this file.

// US-001: per-model default parameter values, sourced from each pipeline's run() signature.
// Fields are optional; absent means the parameter is not applicable for that model.
export interface ModelDefaults {
  width?: number;
  height?: number;
  length?: number;
  fps?: number;
  steps?: number;
  cfg?: number;
  bpm?: number;
}

// Keyed as MODEL_DEFAULTS[media][model].
const MODEL_DEFAULTS: Record<string, Record<string, ModelDefaults>> = {
  image: {
    sdxl:    { width: 1024, height: 1024, steps: 25, cfg: 7.5 },
    anima:   { width: 1024, height: 1024, steps: 30, cfg: 4.0 },
    z_image: { width: 1024, height: 1024, steps: 4 },
  },
  video: {
    ltx2:  { width: 1280, height: 720,  length: 97, fps: 24, steps: 20, cfg: 4.0 },
    ltx23: { width: 768,  height: 512,  length: 97, fps: 25,             cfg: 1.0 },
    wan21: { width: 832,  height: 480,  length: 33, fps: 16, steps: 30, cfg: 6.0 },
    wan22: { width: 832,  height: 480,  length: 81,          steps: 4,  cfg: 1.0 },
  },
  audio: {
    ace_step: { length: 120, steps: 8, cfg: 1.0, bpm: 120 },
  },
};

export function getModelDefaults(media: string, model: string): ModelDefaults | undefined {
  return MODEL_DEFAULTS[media]?.[model];
}

export const MODELS: Record<string, string[]> = {
  "create image": ["sdxl", "anima", "z_image", "flux_klein", "qwen"],
  "create video": ["ltx2", "ltx23", "wan21", "wan22"],
  "create audio": ["ace_step"],
  "edit image":   ["qwen"],
  "edit video":   ["wan21", "wan22"],
};

export const IMAGE_SCRIPTS: Partial<Record<string, string>> = {
  sdxl:    "runtime/image/generation/sdxl/t2i.py",
  anima:   "runtime/image/generation/anima/t2i.py",
  z_image: "runtime/image/generation/z_image/turbo.py",
};

export interface ModelConfig {
  t2v: string;
  i2v?: string;
  cfgFlag: string;
  omitSteps?: true;
}

export const VIDEO_MODEL_CONFIG: Record<string, ModelConfig> = {
  ltx2:  { t2v: "runtime/video/ltx/ltx2/t2v.py",  i2v: "runtime/video/ltx/ltx2/i2v.py",  cfgFlag: "--cfg-pass1" },
  ltx23: { t2v: "runtime/video/ltx/ltx23/t2v.py", i2v: "runtime/video/ltx/ltx23/i2v.py", cfgFlag: "--cfg", omitSteps: true },
  wan21: { t2v: "runtime/video/wan/wan21/t2v.py",  i2v: "runtime/video/wan/wan21/i2v.py",  cfgFlag: "--cfg" },
  wan22: { t2v: "runtime/video/wan/wan22/t2v.py",  i2v: "runtime/video/wan/wan22/i2v.py",  cfgFlag: "--cfg" },
};

export const AUDIO_SCRIPTS: Partial<Record<string, string>> = {
  ace_step: "runtime/audio/ace/t2a.py",
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
