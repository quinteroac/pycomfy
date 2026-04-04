// Tests for US-003: wait_for_job MCP tool.

import { describe, it, expect } from "bun:test";
import { readFileSync } from "fs";
import { join } from "path";

const SRC = readFileSync(join(import.meta.dir, "../src/index.ts"), "utf-8");

// ── AC01: input schema ────────────────────────────────────────────────────────

describe("US-003-AC01: wait_for_job registered with correct schema", () => {
  it("registers a tool named wait_for_job", () => {
    expect(SRC).toContain('"wait_for_job"');
  });

  it("input schema includes job_id as z.string()", () => {
    expect(SRC).toContain("job_id:");
    expect(SRC).toContain("z.string()");
  });

  it("input schema includes timeout_seconds as optional number with default 600", () => {
    expect(SRC).toContain("timeout_seconds:");
    expect(SRC).toContain("z.number()");
    expect(SRC).toContain(".optional()");
    expect(SRC).toContain(".default(600)");
  });
});

// ── AC02: polling logic ───────────────────────────────────────────────────────

describe("US-003-AC02: polls getQueue().getJob every 2 seconds", () => {
  it("calls getQueue", () => {
    expect(SRC).toContain("getQueue()");
  });

  it("calls getJob with the job id", () => {
    expect(SRC).toContain("queue.getJob(input.job_id)");
  });

  it("polls with a 2 second interval", () => {
    expect(SRC).toContain("2000");
  });

  it("checks for completed state", () => {
    expect(SRC).toContain('"completed"');
  });

  it("checks for failed state", () => {
    expect(SRC).toContain('"failed"');
  });
});

// ── AC03: completed result ────────────────────────────────────────────────────

describe("US-003-AC03: on completed returns status, output, duration_seconds", () => {
  it("returns status: completed", () => {
    expect(SRC).toContain('status: "completed"');
  });

  it("returns output field", () => {
    expect(SRC).toContain("output:");
  });

  it("returns duration_seconds field", () => {
    expect(SRC).toContain("duration_seconds:");
  });
});

// ── AC04: failed result ───────────────────────────────────────────────────────

describe("US-003-AC04: on failed returns isError with status and error", () => {
  it("returns isError: true on failure", () => {
    expect(SRC).toContain("isError: true");
  });

  it("returns status: failed", () => {
    expect(SRC).toContain('status: "failed"');
  });

  it("returns error field with failedReason", () => {
    expect(SRC).toContain("failedReason");
  });
});

// ── AC05: timeout result ──────────────────────────────────────────────────────

describe("US-003-AC05: on timeout returns isError with timeout status and message", () => {
  it("returns status: timeout", () => {
    expect(SRC).toContain('status: "timeout"');
  });

  it("includes job_id in timeout response", () => {
    expect(SRC).toContain("job_id: input.job_id");
  });

  it("message mentions timeout seconds and get_job_status", () => {
    expect(SRC).toContain("Use get_job_status to check later.");
  });

  it("deadline uses timeoutSeconds * 1000", () => {
    expect(SRC).toContain("timeoutSeconds * 1000");
  });
});

// ── AC06: queue close ─────────────────────────────────────────────────────────

describe("US-003-AC06: closes queue connection before returning", () => {
  it("calls queue.close()", () => {
    expect(SRC).toContain("queue.close()");
  });

  it("close is in a finally block", () => {
    expect(SRC).toContain("finally");
  });
});

// ── AC07: typecheck ───────────────────────────────────────────────────────────

describe("US-003-AC07: imports getQueue from @parallax/sdk", () => {
  it("imports getQueue from @parallax/sdk", () => {
    expect(SRC).toContain("getQueue");
    expect(SRC).toContain("@parallax/sdk");
  });
});
