// Tests for US-001: create_audio MCP tool — non-blocking job submission.

import { describe, it, expect } from "bun:test";
import { readFileSync } from "fs";
import { join } from "path";

const SRC = readFileSync(join(import.meta.dir, "../src/index.ts"), "utf-8");
const REGISTRY = readFileSync(join(import.meta.dir, "../../parallax_cli/src/models/registry.ts"), "utf-8");

// ── AC01: create_audio calls submitJob ────────────────────────────────────────

describe("US-001-AC01: create_audio calls submitJob", () => {
  it("imports submitJob from @parallax/sdk/submit", () => {
    expect(SRC).toContain("submitJob");
    expect(SRC).toContain("@parallax/sdk/submit");
  });

  it("does not use Bun.spawn subprocess pattern", () => {
    expect(SRC).not.toContain("Bun.spawn");
  });
});

// ── AC02: create_audio returns job_id response ────────────────────────────────

describe("US-001-AC02: create_audio returns job_id response", () => {
  it("response text contains job_id:", () => {
    expect(SRC).toContain("job_id:");
  });

  it("response text contains status: queued", () => {
    expect(SRC).toContain("status: queued");
  });
});

// ── AC03: description mentions job ID ─────────────────────────────────────────

describe("US-001-AC03: create_audio description mentions job ID", () => {
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

describe("US-001 schema: create_audio fields", () => {
  it("registers a tool named create_audio", () => {
    expect(SRC).toContain('"create_audio"');
  });

  it("schema includes required field: model", () => {
    expect(SRC).toContain("model:");
  });

  it("schema includes required field: prompt", () => {
    expect(SRC).toContain("prompt:");
  });

  it("schema includes optional field: length", () => {
    expect(SRC).toContain("length:");
  });

  it("schema includes optional field: steps", () => {
    expect(SRC).toContain("steps:");
  });

  it("schema includes optional field: cfg", () => {
    expect(SRC).toContain("cfg:");
  });

  it("schema includes optional field: bpm", () => {
    expect(SRC).toContain("bpm:");
  });

  it("schema includes optional field: lyrics", () => {
    expect(SRC).toContain("lyrics:");
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

// ── Script registry and arg mapping ───────────────────────────────────────────

describe("US-001 script registry: AUDIO_CREATE_SCRIPTS", () => {
  it("contains ace_step script path", () => {
    expect(REGISTRY).toContain("runtime/audio/ace/t2a.py");
  });

  it("remaps --prompt to --tags for audio scripts", () => {
    expect(SRC).toContain('"--tags"');
  });

  it("remaps --length to --duration for audio scripts", () => {
    expect(SRC).toContain('"--duration"');
  });
});

