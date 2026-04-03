import { describe, it, expect } from "bun:test";
import { buildArgs } from "../../src/models/image";

// US-005-AC01: buildArgs handles standard image args
describe("US-005-AC01: image buildArgs", () => {
  const baseOpts = {
    model: "sdxl",
    prompt: "a cat",
    width: "1024",
    height: "768",
    steps: "20",
    cfg: "7",
    output: "out.png",
  };

  it("includes core flags for a standard model", () => {
    const args = buildArgs(baseOpts, "/models");
    expect(args).toContain("--models-dir");
    expect(args).toContain("/models");
    expect(args).toContain("--prompt");
    expect(args).toContain("a cat");
    expect(args).toContain("--width");
    expect(args).toContain("1024");
    expect(args).toContain("--height");
    expect(args).toContain("768");
    expect(args).toContain("--steps");
    expect(args).toContain("20");
    expect(args).toContain("--output");
    expect(args).toContain("out.png");
  });

  it("includes --cfg for standard models", () => {
    const args = buildArgs(baseOpts, "/models");
    expect(args).toContain("--cfg");
    expect(args).toContain("7");
  });

  it("includes --negative-prompt when provided for standard models", () => {
    const opts = { ...baseOpts, negativePrompt: "blurry" };
    const args = buildArgs(opts, "/models");
    expect(args).toContain("--negative-prompt");
    expect(args).toContain("blurry");
  });

  it("omits --negative-prompt and --cfg for z_image", () => {
    const opts = { ...baseOpts, model: "z_image", negativePrompt: "blurry" };
    const args = buildArgs(opts, "/models");
    expect(args).not.toContain("--negative-prompt");
    expect(args).not.toContain("--cfg");
  });

  it("z_image still receives core flags (prompt, width, height, steps, output)", () => {
    const opts = { ...baseOpts, model: "z_image" };
    const args = buildArgs(opts, "/models");
    expect(args).toContain("--prompt");
    expect(args).toContain("--width");
    expect(args).toContain("--height");
    expect(args).toContain("--steps");
    expect(args).toContain("--output");
  });

  it("includes --seed when provided", () => {
    const opts = { ...baseOpts, seed: "42" };
    const args = buildArgs(opts, "/models");
    expect(args).toContain("--seed");
    expect(args).toContain("42");
  });

  it("omits --seed when not provided", () => {
    const args = buildArgs(baseOpts, "/models");
    expect(args).not.toContain("--seed");
  });
});
