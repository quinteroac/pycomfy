// Tests for US-002: get_job_status MCP tool.

import { describe, it, expect } from "bun:test";
import { readFileSync } from "fs";
import { join } from "path";

const SRC = readFileSync(join(import.meta.dir, "../src/index.ts"), "utf-8");

// ── AC01: input schema ────────────────────────────────────────────────────────

describe("US-002-AC01: get_job_status registered with correct schema", () => {
  it("registers a tool named get_job_status", () => {
    expect(SRC).toContain('"get_job_status"');
  });

  it("input schema includes job_id as z.string()", () => {
    expect(SRC).toContain("job_id:");
    expect(SRC).toContain("z.string()");
  });

  it("imports getJobStatus from @parallax/sdk", () => {
    expect(SRC).toContain("getJobStatus");
    expect(SRC).toContain("@parallax/sdk");
  });
});

// ── AC02: JSON response fields ────────────────────────────────────────────────

describe("US-002-AC02: response includes all required fields", () => {
  it("response includes id field", () => {
    expect(SRC).toContain("id:");
  });

  it("response includes status field", () => {
    expect(SRC).toContain("status:");
  });

  it("response includes progress field", () => {
    expect(SRC).toContain("progress:");
  });

  it("response includes output field", () => {
    expect(SRC).toContain("output:");
  });

  it("response includes error field", () => {
    expect(SRC).toContain("error:");
  });

  it("response includes model field", () => {
    expect(SRC).toContain("model:");
  });

  it("response includes action field", () => {
    expect(SRC).toContain("action:");
  });

  it("response includes media field", () => {
    expect(SRC).toContain("media:");
  });

  it("response is JSON-formatted (JSON.stringify)", () => {
    expect(SRC).toContain("JSON.stringify");
  });
});

// ── AC03: not found returns isError ──────────────────────────────────────────

describe("US-002-AC03: missing job returns isError with message", () => {
  it("returns isError: true when job not found", () => {
    expect(SRC).toContain("isError: true");
  });

  it("error message contains 'not found'", () => {
    expect(SRC).toContain("not found");
  });

  it("error message includes the job id", () => {
    expect(SRC).toContain("input.job_id");
  });
});
