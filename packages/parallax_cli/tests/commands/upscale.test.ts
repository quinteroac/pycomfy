// Tests for US-003: `upscale image` action wires up esrgan and latent_upscale.
// Strategy: call buildUpscaleImageArgs directly and assert the resulting args array
// contains the correct flags and values per model.

import { describe, it, expect } from "bun:test";
import { buildUpscaleImageArgs, type UpscaleImageOpts } from "../../src/models/image";
import { getScript, getModels } from "../../src/models/registry";

// Base opts shared across tests — override per-test as needed.
function baseOpts(model: string, overrides: Partial<UpscaleImageOpts> = {}): UpscaleImageOpts {
  return {
    model,
    prompt:     "test prompt",
    checkpoint: "base.safetensors",
    width:      "768",
    height:     "768",
    steps:      "20",
    cfg:        "7",
    output:     "output.png",
    outputBase: "output_base.png",
    ...overrides,
  };
}

// US-003-AC01: upscale.ts exports registerUpscale (verified via import resolution in typecheck).

// US-003-AC03/AC04: CLI options map to correct args.

describe("US-003-AC04: esrgan args", () => {
  const args = buildUpscaleImageArgs(
    baseOpts("esrgan", { esrganCheckpoint: "RealESRGAN_x4plus.safetensors" }),
    "/models"
  );

  it("includes --models-dir", () => {
    expect(args).toContain("--models-dir");
    expect(args[args.indexOf("--models-dir") + 1]).toBe("/models");
  });

  it("includes --checkpoint", () => {
    expect(args).toContain("--checkpoint");
    expect(args[args.indexOf("--checkpoint") + 1]).toBe("base.safetensors");
  });

  it("includes --prompt", () => {
    expect(args).toContain("--prompt");
    expect(args[args.indexOf("--prompt") + 1]).toBe("test prompt");
  });

  it("includes --width and --height", () => {
    expect(args).toContain("--width");
    expect(args[args.indexOf("--width") + 1]).toBe("768");
    expect(args).toContain("--height");
    expect(args[args.indexOf("--height") + 1]).toBe("768");
  });

  it("includes --steps", () => {
    expect(args).toContain("--steps");
    expect(args[args.indexOf("--steps") + 1]).toBe("20");
  });

  it("includes --cfg", () => {
    expect(args).toContain("--cfg");
    expect(args[args.indexOf("--cfg") + 1]).toBe("7");
  });

  it("includes --output", () => {
    expect(args).toContain("--output");
    expect(args[args.indexOf("--output") + 1]).toBe("output.png");
  });

  it("includes --output-base", () => {
    expect(args).toContain("--output-base");
    expect(args[args.indexOf("--output-base") + 1]).toBe("output_base.png");
  });

  it("includes --esrgan-checkpoint for esrgan", () => {
    expect(args).toContain("--esrgan-checkpoint");
    expect(args[args.indexOf("--esrgan-checkpoint") + 1]).toBe("RealESRGAN_x4plus.safetensors");
  });

  it("does not include --latent-upscale-checkpoint", () => {
    expect(args).not.toContain("--latent-upscale-checkpoint");
  });
});

describe("US-003-AC04: esrgan without esrgan-checkpoint does not add flag", () => {
  const args = buildUpscaleImageArgs(baseOpts("esrgan"), "/models");

  it("omits --esrgan-checkpoint when not provided", () => {
    expect(args).not.toContain("--esrgan-checkpoint");
  });
});

describe("US-003-AC04: latent_upscale args", () => {
  const args = buildUpscaleImageArgs(
    baseOpts("latent_upscale", { latentUpscaleCheckpoint: "4x-UltraSharp.safetensors" }),
    "/models"
  );

  it("includes --models-dir", () => {
    expect(args).toContain("--models-dir");
  });

  it("includes --checkpoint", () => {
    expect(args).toContain("--checkpoint");
  });

  it("includes --latent-upscale-checkpoint for latent_upscale", () => {
    expect(args).toContain("--latent-upscale-checkpoint");
    expect(args[args.indexOf("--latent-upscale-checkpoint") + 1]).toBe("4x-UltraSharp.safetensors");
  });

  it("does not include --esrgan-checkpoint", () => {
    expect(args).not.toContain("--esrgan-checkpoint");
  });
});

describe("US-003-AC04: latent_upscale without checkpoint does not add flag", () => {
  const args = buildUpscaleImageArgs(baseOpts("latent_upscale"), "/models");

  it("omits --latent-upscale-checkpoint when not provided", () => {
    expect(args).not.toContain("--latent-upscale-checkpoint");
  });
});

describe("US-003-AC04: optional flags", () => {
  it("includes --negative-prompt when provided", () => {
    const args = buildUpscaleImageArgs(baseOpts("esrgan", { negativePrompt: "blurry" }), "/models");
    expect(args).toContain("--negative-prompt");
    expect(args[args.indexOf("--negative-prompt") + 1]).toBe("blurry");
  });

  it("omits --negative-prompt when not provided", () => {
    const args = buildUpscaleImageArgs(baseOpts("esrgan"), "/models");
    expect(args).not.toContain("--negative-prompt");
  });

  it("includes --seed when provided", () => {
    const args = buildUpscaleImageArgs(baseOpts("esrgan", { seed: "42" }), "/models");
    expect(args).toContain("--seed");
    expect(args[args.indexOf("--seed") + 1]).toBe("42");
  });

  it("omits --seed when not provided", () => {
    const args = buildUpscaleImageArgs(baseOpts("esrgan"), "/models");
    expect(args).not.toContain("--seed");
  });
});

// US-003-AC05/AC06: getScript returns correct script for each model.
describe("US-003: getScript returns correct upscale scripts", () => {
  it("returns esrgan_upscale.py for esrgan", () => {
    expect(getScript("upscale", "image", "esrgan")).toBe("runtime/image/edit/sd/esrgan_upscale.py");
  });

  it("returns latent_upscale.py for latent_upscale", () => {
    expect(getScript("upscale", "image", "latent_upscale")).toBe("runtime/image/edit/sd/latent_upscale.py");
  });
});

// US-003-AC03: known models list
describe("US-003-AC03: getModels returns upscale image models", () => {
  it("includes esrgan and latent_upscale", () => {
    const models = getModels("upscale", "image");
    expect(models).toContain("esrgan");
    expect(models).toContain("latent_upscale");
  });
});
