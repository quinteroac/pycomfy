import { describe, it, expect } from "bun:test";
import { buildArgs } from "./video";

// US-005-AC02: buildArgs handles video-specific special cases
describe("US-005-AC02: video buildArgs", () => {
  const baseOpts = {
    model: "wan21",
    prompt: "a sunset",
    width: "832",
    height: "480",
    length: "81",
    steps: "30",
    cfg: "6",
    output: "out.mp4",
  };

  it("includes core flags", () => {
    const args = buildArgs(baseOpts, "/models");
    expect(args).toContain("--models-dir");
    expect(args).toContain("/models");
    expect(args).toContain("--prompt");
    expect(args).toContain("a sunset");
    expect(args).toContain("--width");
    expect(args).toContain("832");
    expect(args).toContain("--height");
    expect(args).toContain("480");
    expect(args).toContain("--length");
    expect(args).toContain("81");
    expect(args).toContain("--output");
    expect(args).toContain("out.mp4");
  });

  it("includes --steps for non-distilled models (wan21)", () => {
    const args = buildArgs(baseOpts, "/models");
    expect(args).toContain("--steps");
    expect(args).toContain("30");
  });

  it("omits --steps for distilled models (ltx23, omitSteps: true)", () => {
    const opts = { ...baseOpts, model: "ltx23" };
    const args = buildArgs(opts, "/models");
    expect(args).not.toContain("--steps");
  });

  it("uses --cfg for standard models (wan21)", () => {
    const args = buildArgs(baseOpts, "/models");
    expect(args).toContain("--cfg");
    expect(args).toContain("6");
  });

  it("uses --cfg-pass1 for ltx2 (cfgFlag override)", () => {
    const opts = { ...baseOpts, model: "ltx2" };
    const args = buildArgs(opts, "/models");
    expect(args).toContain("--cfg-pass1");
    expect(args).not.toContain("--cfg");
  });

  it("maps --input to --image when model supports i2v", () => {
    const opts = { ...baseOpts, input: "/tmp/frame.png" };
    const args = buildArgs(opts, "/models");
    expect(args).toContain("--image");
    expect(args).toContain("/tmp/frame.png");
    expect(args).not.toContain("--input");
  });

  it("does not add --image when no --input is provided (t2v path)", () => {
    const args = buildArgs(baseOpts, "/models");
    expect(args).not.toContain("--image");
    expect(args).not.toContain("--input");
  });

  it("includes --seed when provided", () => {
    const opts = { ...baseOpts, seed: "99" };
    const args = buildArgs(opts, "/models");
    expect(args).toContain("--seed");
    expect(args).toContain("99");
  });

  it("omits --seed when not provided", () => {
    const args = buildArgs(baseOpts, "/models");
    expect(args).not.toContain("--seed");
  });
});
