// Tests for US-001-AC01–AC05: create_image MCP tool — non-blocking job submission.

import { describe, it, expect } from "bun:test";
import { readFileSync } from "fs";
import { join } from "path";

const SRC = readFileSync(join(import.meta.dir, "../src/index.ts"), "utf-8");

// ── AC01: create_image calls submitJob from @parallax/sdk/submit ──────────────

describe("US-001-AC01: create_image calls submitJob", () => {
  it("imports submitJob from @parallax/sdk/submit", () => {
    expect(SRC).toContain("submitJob");
    expect(SRC).toContain("@parallax/sdk/submit");
  });

  it("does not use Bun.spawn subprocess pattern", () => {
    expect(SRC).not.toContain("Bun.spawn");
  });
});

// ── AC02: create_image returns job_id response immediately ────────────────────

describe("US-001-AC02: create_image returns job_id response", () => {
  it("response contains job_id:", () => {
    expect(SRC).toContain("job_id:");
  });

  it("response contains status: queued", () => {
    expect(SRC).toContain("status: queued");
  });

  it("response contains model:", () => {
    expect(SRC).toContain("model: ${input.model}");
  });
});

// ── AC03: tool descriptions mention job ID ────────────────────────────────────

describe("US-001-AC03: create_image description mentions job ID", () => {
  it("description mentions Returns a job ID", () => {
    expect(SRC).toContain("Returns a job ID immediately");
  });
});

// ── AC04: Bun.spawn subprocess pattern is removed ────────────────────────────

describe("US-001-AC04: Bun.spawn removed from index.ts", () => {
  it("Bun.spawn is not present anywhere in index.ts", () => {
    expect(SRC).not.toContain("Bun.spawn");
  });

  it("CLI_DIR is not present (no subprocess CLI invocation)", () => {
    expect(SRC).not.toContain("CLI_DIR");
  });
});

// ── Schema fields still present ───────────────────────────────────────────────

describe("US-001 schema: create_image fields", () => {
  it("registers a tool named create_image", () => {
    expect(SRC).toContain('"create_image"');
  });

  it("schema includes required field: model", () => {
    expect(SRC).toContain("model:");
  });

  it("schema includes required field: prompt", () => {
    expect(SRC).toContain("prompt:");
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

  it("schema includes optional field: modelsDir", () => {
    expect(SRC).toContain("modelsDir:");
  });
});

// ── Script registry ───────────────────────────────────────────────────────────

describe("US-001 script registry: IMAGE_CREATE_SCRIPTS", () => {
  it("contains sdxl script path", () => {
    expect(SRC).toContain("runtime/image/generation/sdxl/t2i.py");
  });

  it("contains anima script path", () => {
    expect(SRC).toContain("runtime/image/generation/anima/t2i.py");
  });

  it("contains z_image script path", () => {
    expect(SRC).toContain("runtime/image/generation/z_image/turbo.py");
  });
});

