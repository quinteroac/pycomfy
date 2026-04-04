// Tests for US-004: Update tool descriptions for agent discoverability.

import { describe, it, expect } from "bun:test";
import { readFileSync } from "fs";
import { join } from "path";

const SRC = readFileSync(join(import.meta.dir, "../src/index.ts"), "utf-8");

const INFERENCE_TOOL_SUFFIX =
  "Returns a job_id. Use get_job_status to poll or wait_for_job to block until done.";

// ── AC01: inference tool descriptions end with the canonical suffix ───────────

describe("US-004-AC01: inference tool descriptions end with canonical suffix", () => {
  it("create_image description ends with the canonical suffix", () => {
    expect(SRC).toContain(
      `Generate an image using the Parallax pipeline (parallax create image). ${INFERENCE_TOOL_SUFFIX}`
    );
  });

  it("create_video description ends with the canonical suffix", () => {
    expect(SRC).toContain(
      `Generate a video using the Parallax pipeline (parallax create video). ${INFERENCE_TOOL_SUFFIX}`
    );
  });

  it("create_audio description ends with the canonical suffix", () => {
    expect(SRC).toContain(
      `Generate audio using the Parallax pipeline (parallax create audio). ${INFERENCE_TOOL_SUFFIX}`
    );
  });

  it("edit_image description ends with the canonical suffix", () => {
    expect(SRC).toContain(
      `Edit an image using the Parallax pipeline (parallax edit image). ${INFERENCE_TOOL_SUFFIX}`
    );
  });

  it("upscale_image description ends with the canonical suffix", () => {
    expect(SRC).toContain(
      `Upscale an image using the Parallax pipeline (parallax upscale image). ${INFERENCE_TOOL_SUFFIX}`
    );
  });
});

// ── AC02: get_job_status description ─────────────────────────────────────────

describe("US-004-AC02: get_job_status description is exact", () => {
  it("get_job_status has the required description", () => {
    expect(SRC).toContain(
      "Check the current status and progress of a submitted inference job. Returns status, progress percentage (0-100), and output path when completed."
    );
  });
});

// ── AC03: wait_for_job description ───────────────────────────────────────────

describe("US-004-AC03: wait_for_job description is exact", () => {
  it("wait_for_job has the required description", () => {
    expect(SRC).toContain(
      "Block until a submitted inference job completes. Polls internally every 2 seconds. Default timeout: 600 seconds. Returns output path on success."
    );
  });
});
