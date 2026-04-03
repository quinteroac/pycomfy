import { describe, it, expect } from "bun:test";
import {
  MODELS,
  IMAGE_SCRIPTS,
  VIDEO_MODEL_CONFIG,
  AUDIO_SCRIPTS,
  getModels,
  getScript,
  getModelConfig,
  type ModelConfig,
} from "./registry";

// US-004-AC01: MODELS, IMAGE_SCRIPTS, VIDEO_MODEL_CONFIG, AUDIO_SCRIPTS are exported
describe("US-004-AC01: registry exports data constants", () => {
  it("MODELS contains expected action+media keys", () => {
    expect(MODELS["create image"]).toEqual(["sdxl", "anima", "z_image", "flux_klein", "qwen"]);
    expect(MODELS["create video"]).toEqual(["ltx2", "ltx23", "wan21", "wan22"]);
    expect(MODELS["create audio"]).toEqual(["ace_step"]);
    expect(MODELS["edit image"]).toEqual(["qwen"]);
    expect(MODELS["edit video"]).toEqual(["wan21", "wan22"]);
  });

  it("IMAGE_SCRIPTS maps known image models to script paths", () => {
    expect(IMAGE_SCRIPTS["sdxl"]).toBe("examples/image/generation/sdxl/t2i.py");
    expect(IMAGE_SCRIPTS["anima"]).toBe("examples/image/generation/anima/t2i.py");
    expect(IMAGE_SCRIPTS["z_image"]).toBe("examples/image/generation/z_image/turbo.py");
    expect(IMAGE_SCRIPTS["flux_klein"]).toBeUndefined();
  });

  it("VIDEO_MODEL_CONFIG maps known video models to config objects", () => {
    const ltx2 = VIDEO_MODEL_CONFIG["ltx2"];
    expect(ltx2.t2v).toContain("ltx2");
    expect(ltx2.cfgFlag).toBe("--cfg-pass1");
    const ltx23 = VIDEO_MODEL_CONFIG["ltx23"];
    expect(ltx23.omitSteps).toBe(true);
    expect(ltx23.cfgFlag).toBe("--cfg");
    expect(VIDEO_MODEL_CONFIG["wan21"].cfgFlag).toBe("--cfg");
    expect(VIDEO_MODEL_CONFIG["wan22"].cfgFlag).toBe("--cfg");
  });

  it("AUDIO_SCRIPTS maps ace_step to a script path", () => {
    expect(AUDIO_SCRIPTS["ace_step"]).toBe("examples/audio/ace/t2a.py");
  });
});

// US-004-AC02: getModels, getScript, getModelConfig are exported with correct signatures
describe("US-004-AC02: getModels(action, media)", () => {
  it("returns the correct model list for create image", () => {
    expect(getModels("create", "image")).toEqual(["sdxl", "anima", "z_image", "flux_klein", "qwen"]);
  });

  it("returns the correct model list for create video", () => {
    expect(getModels("create", "video")).toEqual(["ltx2", "ltx23", "wan21", "wan22"]);
  });

  it("returns the correct model list for edit image", () => {
    expect(getModels("edit", "image")).toEqual(["qwen"]);
  });

  it("returns empty array for unknown action+media", () => {
    expect(getModels("delete", "image")).toEqual([]);
  });
});

describe("US-004-AC02: getScript(action, media, model)", () => {
  it("returns script path for a known image model", () => {
    expect(getScript("create", "image", "sdxl")).toBe("examples/image/generation/sdxl/t2i.py");
  });

  it("returns undefined for an image model without a script (not yet implemented)", () => {
    expect(getScript("create", "image", "flux_klein")).toBeUndefined();
  });

  it("returns t2v script path for a known video model", () => {
    const script = getScript("create", "video", "ltx2");
    expect(script).toBe("examples/video/ltx/ltx2/t2v.py");
  });

  it("returns undefined for an unknown video model", () => {
    expect(getScript("create", "video", "unknown_model")).toBeUndefined();
  });

  it("returns script path for a known audio model", () => {
    expect(getScript("create", "audio", "ace_step")).toBe("examples/audio/ace/t2a.py");
  });

  it("returns undefined for an unknown audio model", () => {
    expect(getScript("create", "audio", "bark")).toBeUndefined();
  });
});

describe("US-004-AC02: getModelConfig(media, model)", () => {
  it("returns a ModelConfig for a known video model", () => {
    const config: ModelConfig | undefined = getModelConfig("video", "wan21");
    expect(config).toBeDefined();
    expect(config?.cfgFlag).toBe("--cfg");
    expect(config?.i2v).toContain("i2v.py");
  });

  it("returns ModelConfig with omitSteps for distilled models (ltx23)", () => {
    const config = getModelConfig("video", "ltx23");
    expect(config?.omitSteps).toBe(true);
  });

  it("returns --cfg-pass1 cfgFlag for ltx2", () => {
    const config = getModelConfig("video", "ltx2");
    expect(config?.cfgFlag).toBe("--cfg-pass1");
  });

  it("returns undefined for an unknown video model", () => {
    expect(getModelConfig("video", "unknown")).toBeUndefined();
  });

  it("returns undefined for non-video media (image has no per-model config)", () => {
    expect(getModelConfig("image", "sdxl")).toBeUndefined();
  });

  it("returns undefined for non-video media (audio)", () => {
    expect(getModelConfig("audio", "ace_step")).toBeUndefined();
  });
});
