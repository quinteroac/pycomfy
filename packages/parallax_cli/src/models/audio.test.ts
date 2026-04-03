import { describe, it, expect } from "bun:test";
import { buildArgs } from "./audio";

// US-005-AC03: buildArgs handles audio flag remapping
describe("US-005-AC03: audio buildArgs", () => {
  const baseOpts = {
    model: "ace_step",
    prompt: "upbeat jazz",
    length: "30",
    steps: "60",
    cfg: "2",
    bpm: "120",
    lyrics: "",
    output: "out.wav",
  };

  it("includes core flags", () => {
    const args = buildArgs(baseOpts, "/models");
    expect(args).toContain("--models-dir");
    expect(args).toContain("/models");
    expect(args).toContain("--steps");
    expect(args).toContain("60");
    expect(args).toContain("--cfg");
    expect(args).toContain("2");
    expect(args).toContain("--bpm");
    expect(args).toContain("120");
    expect(args).toContain("--output");
    expect(args).toContain("out.wav");
  });

  it("maps --prompt to --tags", () => {
    const args = buildArgs(baseOpts, "/models");
    expect(args).toContain("--tags");
    expect(args).toContain("upbeat jazz");
    expect(args).not.toContain("--prompt");
  });

  it("maps --length to --duration", () => {
    const args = buildArgs(baseOpts, "/models");
    expect(args).toContain("--duration");
    expect(args).toContain("30");
    expect(args).not.toContain("--length");
  });

  it("includes --lyrics flag", () => {
    const opts = { ...baseOpts, lyrics: "verse one" };
    const args = buildArgs(opts, "/models");
    expect(args).toContain("--lyrics");
    expect(args).toContain("verse one");
  });

  it("includes --seed when provided", () => {
    const opts = { ...baseOpts, seed: "7" };
    const args = buildArgs(opts, "/models");
    expect(args).toContain("--seed");
    expect(args).toContain("7");
  });

  it("omits --seed when not provided", () => {
    const args = buildArgs(baseOpts, "/models");
    expect(args).not.toContain("--seed");
  });
});
