import { describe, it, expect } from "bun:test";
import {
  MODELS,
  IMAGE_SCRIPTS,
  VIDEO_MODEL_CONFIG,
  AUDIO_SCRIPTS,
  getModels,
  getScript,
  getModelConfig,
  getModelDefaults,
  type ModelConfig,
  type ModelDefaults,
} from "../../src/models/registry";

// US-001-AC01: ModelDefaults interface has the correct optional fields
describe("US-001-AC01: ModelDefaults interface", () => {
  it("accepts a fully-populated object", () => {
    const d: ModelDefaults = { width: 1024, height: 1024, length: 97, fps: 24, steps: 20, cfg: 4.0 };
    expect(d.width).toBe(1024);
    expect(d.fps).toBe(24);
  });

  it("accepts a partial object (absent fields are undefined)", () => {
    const d: ModelDefaults = { steps: 4 };
    expect(d.width).toBeUndefined();
    expect(d.steps).toBe(4);
  });
});

// US-001-AC02: every model with a create script has a ModelDefaults entry
describe("US-001-AC02: per-model defaults populated from run() signatures", () => {
  it("sdxl defaults match its run() signature", () => {
    const d = getModelDefaults("image", "sdxl");
    expect(d).toBeDefined();
    expect(d?.width).toBe(1024);
    expect(d?.height).toBe(1024);
    expect(d?.steps).toBe(25);
    expect(d?.cfg).toBe(7.5);
  });

  it("anima defaults match its run() signature", () => {
    const d = getModelDefaults("image", "anima");
    expect(d).toBeDefined();
    expect(d?.width).toBe(1024);
    expect(d?.height).toBe(1024);
    expect(d?.steps).toBe(30);
    expect(d?.cfg).toBe(4.0);
  });

  it("z_image defaults match its run() signature (no cfg)", () => {
    const d = getModelDefaults("image", "z_image");
    expect(d).toBeDefined();
    expect(d?.width).toBe(1024);
    expect(d?.height).toBe(1024);
    expect(d?.steps).toBe(4);
    expect(d?.cfg).toBeUndefined();
  });

  it("ltx2 defaults match its run() signature", () => {
    const d = getModelDefaults("video", "ltx2");
    expect(d).toBeDefined();
    expect(d?.width).toBe(1280);
    expect(d?.height).toBe(720);
    expect(d?.length).toBe(97);
    expect(d?.fps).toBe(24);
    expect(d?.steps).toBe(20);
    expect(d?.cfg).toBe(4.0);
  });

  it("ltx23 defaults match its run() signature (no steps — distilled)", () => {
    const d = getModelDefaults("video", "ltx23");
    expect(d).toBeDefined();
    expect(d?.width).toBe(768);
    expect(d?.height).toBe(512);
    expect(d?.length).toBe(97);
    expect(d?.fps).toBe(25);
    expect(d?.steps).toBeUndefined();
    expect(d?.cfg).toBe(1.0);
  });

  it("wan21 defaults match its run() signature", () => {
    const d = getModelDefaults("video", "wan21");
    expect(d).toBeDefined();
    expect(d?.width).toBe(832);
    expect(d?.height).toBe(480);
    expect(d?.length).toBe(33);
    expect(d?.fps).toBe(16);
    expect(d?.steps).toBe(30);
    expect(d?.cfg).toBe(6.0);
  });

  it("wan22 defaults match its run() signature", () => {
    const d = getModelDefaults("video", "wan22");
    expect(d).toBeDefined();
    expect(d?.width).toBe(832);
    expect(d?.height).toBe(480);
    expect(d?.length).toBe(81);
    expect(d?.steps).toBe(4);
    expect(d?.cfg).toBe(1.0);
  });

  it("ace_step defaults match its run() signature", () => {
    const d = getModelDefaults("audio", "ace_step");
    expect(d).toBeDefined();
    expect(d?.length).toBe(120);
    expect(d?.steps).toBe(8);
    expect(d?.cfg).toBe(1.0);
    expect(d?.width).toBeUndefined();
    expect(d?.height).toBeUndefined();
  });
});

// US-001-AC03: getModelDefaults(media, model) is exported and returns ModelDefaults | undefined
describe("US-001-AC03: getModelDefaults helper", () => {
  it("returns ModelDefaults for a known image model", () => {
    const d = getModelDefaults("image", "sdxl");
    expect(d).toBeDefined();
  });

  it("returns ModelDefaults for a known video model", () => {
    const d = getModelDefaults("video", "wan21");
    expect(d).toBeDefined();
  });

  it("returns ModelDefaults for a known audio model", () => {
    const d = getModelDefaults("audio", "ace_step");
    expect(d).toBeDefined();
  });

  it("returns undefined for an unknown model", () => {
    expect(getModelDefaults("image", "unknown_model")).toBeUndefined();
  });

  it("returns undefined for an unknown media type", () => {
    expect(getModelDefaults("video3d", "sdxl")).toBeUndefined();
  });

  it("returns undefined for models without a create script (flux_klein, qwen)", () => {
    expect(getModelDefaults("image", "flux_klein")).toBeUndefined();
    expect(getModelDefaults("image", "qwen")).toBeUndefined();
  });
});

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
    expect(IMAGE_SCRIPTS["sdxl"]).toBe("runtime/image/generation/sdxl/t2i.py");
    expect(IMAGE_SCRIPTS["anima"]).toBe("runtime/image/generation/anima/t2i.py");
    expect(IMAGE_SCRIPTS["z_image"]).toBe("runtime/image/generation/z_image/turbo.py");
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
    expect(AUDIO_SCRIPTS["ace_step"]).toBe("runtime/audio/ace/t2a.py");
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
    expect(getScript("create", "image", "sdxl")).toBe("runtime/image/generation/sdxl/t2i.py");
  });

  it("returns undefined for an image model without a script (not yet implemented)", () => {
    expect(getScript("create", "image", "flux_klein")).toBeUndefined();
  });

  it("returns t2v script path for a known video model", () => {
    const script = getScript("create", "video", "ltx2");
    expect(script).toBe("runtime/video/ltx/ltx2/t2v.py");
  });

  it("returns undefined for an unknown video model", () => {
    expect(getScript("create", "video", "unknown_model")).toBeUndefined();
  });

  it("returns script path for a known audio model", () => {
    expect(getScript("create", "audio", "ace_step")).toBe("runtime/audio/ace/t2a.py");
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
