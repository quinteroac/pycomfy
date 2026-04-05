// Tests for US-001: edit_image MCP tool — non-blocking job submission.

import { describe, it, expect } from "bun:test";
import { readFileSync } from "fs";
import { join } from "path";

const SRC = readFileSync(join(import.meta.dir, "../src/index.ts"), "utf-8");
const REGISTRY = readFileSync(join(import.meta.dir, "../../parallax_cli/src/models/registry.ts"), "utf-8");

// ── AC01: edit_image calls submitJob ─────────────────────────────────────────

describe("US-001-AC01: edit_image calls submitJob", () => {
  it("imports submitJob from @parallax/sdk/submit", () => {
    expect(SRC).toContain("submitJob");
    expect(SRC).toContain("@parallax/sdk/submit");
  });

  it("does not use Bun.spawn subprocess pattern", () => {
    expect(SRC).not.toContain("Bun.spawn");
  });
});

// ── AC02: edit_image returns job_id response ──────────────────────────────────

describe("US-001-AC02: edit_image returns job_id response", () => {
  it("response text contains job_id:", () => {
    expect(SRC).toContain("job_id:");
  });

  it("response text contains status: queued", () => {
    expect(SRC).toContain("status: queued");
  });
});

// ── AC03: description mentions job ID ─────────────────────────────────────────

describe("US-001-AC03: edit_image description mentions job ID", () => {
  it("description says Returns a job_id", () => {
    expect(SRC).toContain("Returns a job_id. Use get_job_status to poll or wait_for_job to block until done.");
  });
});

// ── AC04: Bun.spawn is removed ────────────────────────────────────────────────

describe("US-001-AC04: Bun.spawn removed", () => {
  it("Bun.spawn is not present in index.ts", () => {
    expect(SRC).not.toContain("Bun.spawn");
  });
});

// ── Schema fields ─────────────────────────────────────────────────────────────

describe("US-001 schema: edit_image fields", () => {
  it("registers a tool named edit_image", () => {
    expect(SRC).toContain('"edit_image"');
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

  it("schema includes optional field: subjectImage", () => {
    expect(SRC).toContain("subjectImage:");
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

  it("schema includes optional field: image2", () => {
    expect(SRC).toContain("image2:");
  });

  it("schema includes optional field: image3", () => {
    expect(SRC).toContain("image3:");
  });

  it("schema includes optional field: noLora", () => {
    expect(SRC).toContain("noLora:");
  });

  it("schema includes optional field: modelsDir", () => {
    expect(SRC).toContain("modelsDir:");
  });
});

// ── Script registry and model-specific logic ──────────────────────────────────

describe("US-001 script registry: IMAGE_EDIT_SCRIPTS", () => {
  it("contains flux_9b_kv script path", () => {
    expect(REGISTRY).toContain("runtime/image/edit/flux/9b_kv.py");
  });

  it("contains qwen edit_2511 script path", () => {
    expect(REGISTRY).toContain("runtime/image/edit/qwen/edit_2511.py");
  });

  it("handles qwen model with --output-prefix", () => {
    expect(SRC).toContain('"--output-prefix"');
  });

  it("handles flux_9b_kv model with --subject-image", () => {
    expect(SRC).toContain('"--subject-image"');
  });
});

