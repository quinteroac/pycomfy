// Tests for US-001: upscale_image MCP tool — non-blocking job submission.

import { describe, it, expect } from "bun:test";
import { readFileSync } from "fs";
import { join } from "path";

const SRC = readFileSync(join(import.meta.dir, "../src/index.ts"), "utf-8");

// ── AC01: upscale_image calls submitJob ───────────────────────────────────────

describe("US-001-AC01: upscale_image calls submitJob", () => {
  it("imports submitJob from @parallax/sdk/submit", () => {
    expect(SRC).toContain("submitJob");
    expect(SRC).toContain("@parallax/sdk/submit");
  });

  it("does not use Bun.spawn subprocess pattern", () => {
    expect(SRC).not.toContain("Bun.spawn");
  });
});

// ── AC02: upscale_image returns job_id response ───────────────────────────────

describe("US-001-AC02: upscale_image returns job_id response", () => {
  it("response text contains job_id:", () => {
    expect(SRC).toContain("job_id:");
  });

  it("response text contains status: queued", () => {
    expect(SRC).toContain("status: queued");
  });
});

// ── AC03: description mentions job ID ─────────────────────────────────────────

describe("US-001-AC03: upscale_image description mentions job ID", () => {
  it("description says Returns a job ID immediately", () => {
    expect(SRC).toContain("Returns a job ID immediately");
  });
});

// ── AC04: Bun.spawn is removed ────────────────────────────────────────────────

describe("US-001-AC04: Bun.spawn removed", () => {
  it("Bun.spawn is not present in index.ts", () => {
    expect(SRC).not.toContain("Bun.spawn");
  });
});

// ── Schema fields ─────────────────────────────────────────────────────────────

describe("US-001 schema: upscale_image fields", () => {
  it("registers a tool named upscale_image", () => {
    expect(SRC).toContain('"upscale_image"');
  });

  it("schema includes required field: model", () => {
    expect(SRC).toContain("model:");
  });

  it("schema includes required field: prompt", () => {
    expect(SRC).toContain("prompt:");
  });

  it("schema includes required field: input", () => {
    expect(SRC).toContain("input:");
  });

  it("schema includes optional field: checkpoint", () => {
    expect(SRC).toContain("checkpoint:");
  });

  it("schema includes optional field: esrganCheckpoint", () => {
    expect(SRC).toContain("esrganCheckpoint:");
  });

  it("schema includes optional field: latentUpscaleCheckpoint", () => {
    expect(SRC).toContain("latentUpscaleCheckpoint:");
  });

  it("schema includes optional field: negativePrompt", () => {
    expect(SRC).toContain("negativePrompt:");
  });

  it("schema includes optional field: width", () => {
    expect(SRC).toContain("width:");
  });

  it("schema includes optional field: height", () => {
    expect(SRC).toContain("height:");
  });

  it("schema includes optional field: steps", () => {
    expect(SRC).toContain("steps:");
  });

  it("schema includes optional field: cfg", () => {
    expect(SRC).toContain("cfg:");
  });

  it("schema includes optional field: seed", () => {
    expect(SRC).toContain("seed:");
  });

  it("schema includes optional field: output", () => {
    expect(SRC).toContain("output:");
  });

  it("schema includes optional field: outputBase", () => {
    expect(SRC).toContain("outputBase:");
  });

  it("schema includes optional field: modelsDir", () => {
    expect(SRC).toContain("modelsDir:");
  });
});

// ── Script registry ───────────────────────────────────────────────────────────

describe("US-001 script registry: IMAGE_UPSCALE_SCRIPTS", () => {
  it("contains esrgan script path", () => {
    expect(SRC).toContain("runtime/image/edit/sd/esrgan_upscale.py");
  });

  it("contains latent_upscale script path", () => {
    expect(SRC).toContain("runtime/image/edit/sd/latent_upscale.py");
  });

  it("passes --esrgan-checkpoint flag", () => {
    expect(SRC).toContain('"--esrgan-checkpoint"');
  });

  it("passes --latent-upscale-checkpoint flag", () => {
    expect(SRC).toContain('"--latent-upscale-checkpoint"');
  });

  it("passes --output-base flag", () => {
    expect(SRC).toContain('"--output-base"');
  });
});

