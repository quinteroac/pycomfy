// Tests for US-002: CLI applies per-model defaults at runtime.
// Strategy: simulate what the action handler does — call getModelDefaults then buildArgs —
// and assert the resulting args array contains the correct flags and values.

import { describe, it, expect } from "bun:test";
import { getModelDefaults } from "../../src/models/registry";
import { buildArgs as buildVideoArgs, type VideoOpts } from "../../src/models/video";
import { buildArgs as buildAudioArgs, type AudioOpts } from "../../src/models/audio";

// Helper: merge model defaults into opts the same way the action handler does.
function applyVideoDefaults(
  model: string,
  overrides: Partial<Omit<VideoOpts, "model" | "prompt" | "output">> = {}
): VideoOpts {
  const defaults = getModelDefaults("video", model);
  return {
    model,
    prompt: "test",
    width:  overrides.width  ?? (defaults?.width  != null ? String(defaults.width)  : "832"),
    height: overrides.height ?? (defaults?.height != null ? String(defaults.height) : "480"),
    length: overrides.length ?? (defaults?.length != null ? String(defaults.length) : "81"),
    steps:  overrides.steps  ?? (defaults?.steps  != null ? String(defaults.steps)  : "30"),
    cfg:    overrides.cfg    ?? (defaults?.cfg    != null ? String(defaults.cfg)    : "6"),
    seed:   overrides.seed,
    output: overrides.output ?? "output.mp4",
  };
}

function applyAudioDefaults(
  model: string,
  overrides: Partial<Omit<AudioOpts, "model" | "prompt" | "output" | "bpm" | "lyrics">> = {}
): AudioOpts {
  const defaults = getModelDefaults("audio", model);
  return {
    model,
    prompt:  "test",
    length:  overrides.length ?? (defaults?.length != null ? String(defaults.length) : "30"),
    steps:   overrides.steps  ?? (defaults?.steps  != null ? String(defaults.steps)  : "60"),
    cfg:     overrides.cfg    ?? (defaults?.cfg    != null ? String(defaults.cfg)    : "2"),
    bpm:     "120",
    lyrics:  "",
    seed:    overrides.seed,
    output:  overrides.output ?? "output.wav",
  };
}

// US-002-AC01: static commander defaults are removed (verified structurally by absence of
// default values in .option() calls — enforced by code review, not unit test).

// US-002-AC03: ltx2 defaults applied when user omits flags
describe("US-002-AC03: ltx2 model defaults", () => {
  it("passes --width 1280 to Python script", () => {
    const args = buildVideoArgs(applyVideoDefaults("ltx2"), "/models");
    const idx = args.indexOf("--width");
    expect(idx).toBeGreaterThan(-1);
    expect(args[idx + 1]).toBe("1280");
  });

  it("passes --height 720 to Python script", () => {
    const args = buildVideoArgs(applyVideoDefaults("ltx2"), "/models");
    const idx = args.indexOf("--height");
    expect(idx).toBeGreaterThan(-1);
    expect(args[idx + 1]).toBe("720");
  });

  it("passes --length 97 to Python script", () => {
    const args = buildVideoArgs(applyVideoDefaults("ltx2"), "/models");
    const idx = args.indexOf("--length");
    expect(idx).toBeGreaterThan(-1);
    expect(args[idx + 1]).toBe("97");
  });

  it("passes --steps 20 to Python script", () => {
    const args = buildVideoArgs(applyVideoDefaults("ltx2"), "/models");
    const idx = args.indexOf("--steps");
    expect(idx).toBeGreaterThan(-1);
    expect(args[idx + 1]).toBe("20");
  });

  it("passes --cfg-pass1 4 to Python script", () => {
    const args = buildVideoArgs(applyVideoDefaults("ltx2"), "/models");
    const idx = args.indexOf("--cfg-pass1");
    expect(idx).toBeGreaterThan(-1);
    expect(args[idx + 1]).toBe("4");
  });
});

// US-002-AC04: wan21 defaults applied when user omits flags
describe("US-002-AC04: wan21 model defaults", () => {
  it("passes --width 832 to Python script", () => {
    const args = buildVideoArgs(applyVideoDefaults("wan21"), "/models");
    const idx = args.indexOf("--width");
    expect(idx).toBeGreaterThan(-1);
    expect(args[idx + 1]).toBe("832");
  });

  it("passes --height 480 to Python script", () => {
    const args = buildVideoArgs(applyVideoDefaults("wan21"), "/models");
    const idx = args.indexOf("--height");
    expect(idx).toBeGreaterThan(-1);
    expect(args[idx + 1]).toBe("480");
  });

  it("passes --length 33 to Python script", () => {
    const args = buildVideoArgs(applyVideoDefaults("wan21"), "/models");
    const idx = args.indexOf("--length");
    expect(idx).toBeGreaterThan(-1);
    expect(args[idx + 1]).toBe("33");
  });

  it("passes --steps 30 to Python script", () => {
    const args = buildVideoArgs(applyVideoDefaults("wan21"), "/models");
    const idx = args.indexOf("--steps");
    expect(idx).toBeGreaterThan(-1);
    expect(args[idx + 1]).toBe("30");
  });

  it("passes --cfg 6 to Python script", () => {
    const args = buildVideoArgs(applyVideoDefaults("wan21"), "/models");
    const idx = args.indexOf("--cfg");
    expect(idx).toBeGreaterThan(-1);
    expect(args[idx + 1]).toBe("6");
  });
});

// US-002-AC05: ace_step defaults applied when user omits flags
describe("US-002-AC05: ace_step model defaults", () => {
  it("passes --duration 120 to Python script", () => {
    const args = buildAudioArgs(applyAudioDefaults("ace_step"), "/models");
    const idx = args.indexOf("--duration");
    expect(idx).toBeGreaterThan(-1);
    expect(args[idx + 1]).toBe("120");
  });

  it("passes --steps 8 to Python script", () => {
    const args = buildAudioArgs(applyAudioDefaults("ace_step"), "/models");
    const idx = args.indexOf("--steps");
    expect(idx).toBeGreaterThan(-1);
    expect(args[idx + 1]).toBe("8");
  });

  it("passes --cfg 1 to Python script", () => {
    const args = buildAudioArgs(applyAudioDefaults("ace_step"), "/models");
    const idx = args.indexOf("--cfg");
    expect(idx).toBeGreaterThan(-1);
    expect(args[idx + 1]).toBe("1");
  });
});

// US-002-AC06: explicitly supplied flags override model defaults
describe("US-002-AC06: explicit flags override model defaults", () => {
  it("--steps 50 overrides ltx2 default of 20", () => {
    const args = buildVideoArgs(applyVideoDefaults("ltx2", { steps: "50" }), "/models");
    const idx = args.indexOf("--steps");
    expect(idx).toBeGreaterThan(-1);
    expect(args[idx + 1]).toBe("50");
  });

  it("--width 640 overrides wan21 default of 832", () => {
    const args = buildVideoArgs(applyVideoDefaults("wan21", { width: "640" }), "/models");
    const idx = args.indexOf("--width");
    expect(idx).toBeGreaterThan(-1);
    expect(args[idx + 1]).toBe("640");
  });

  it("--length 60 overrides ace_step default of 120", () => {
    const args = buildAudioArgs(applyAudioDefaults("ace_step", { length: "60" }), "/models");
    const idx = args.indexOf("--duration");
    expect(idx).toBeGreaterThan(-1);
    expect(args[idx + 1]).toBe("60");
  });
});
