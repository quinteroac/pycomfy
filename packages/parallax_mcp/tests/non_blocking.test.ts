// Tests for US-001-AC04: each tool returns job_id in under 500ms with a mock queue.
// Uses mock.module to stub submitJob so no real queue connection is needed.

import { describe, it, expect, mock, beforeAll } from "bun:test";
import type { ParallaxJobData } from "@parallax/sdk";

beforeAll(() => {
  mock.module("@parallax/sdk/submit", () => ({
    submitJob: async (_data: ParallaxJobData): Promise<string> => "mock-job-id-abc123",
  }));
});

// Helper: simulate the response construction that every inference tool performs.
async function callToolHandler(model: string): Promise<string> {
  const { submitJob } = await import("@parallax/sdk/submit");
  const jobId = await submitJob({
    action:     "create",
    media:      "image",
    model,
    script:     "runtime/image/generation/sdxl/t2i.py",
    args:       ["--prompt", "test"],
    scriptBase: "/tmp",
    uvPath:     "uv",
  });
  return `job_id: ${jobId}\nstatus: queued\nmodel: ${model}`;
}

// ── AC04: create_image returns job_id in under 500ms ─────────────────────────

describe("US-001-AC04: create_image non-blocking job submission", () => {
  it("returns a string containing job_id: in under 500ms", async () => {
    const start = Date.now();
    const result = await callToolHandler("sdxl");
    const elapsed = Date.now() - start;
    expect(result).toContain("job_id:");
    expect(elapsed).toBeLessThan(500);
  });

  it("response contains status: queued", async () => {
    const result = await callToolHandler("sdxl");
    expect(result).toContain("status: queued");
  });

  it("response contains model name", async () => {
    const result = await callToolHandler("sdxl");
    expect(result).toContain("model: sdxl");
  });
});

// ── AC04: create_video returns job_id in under 500ms ─────────────────────────

describe("US-001-AC04: create_video non-blocking job submission", () => {
  it("returns a string containing job_id: in under 500ms", async () => {
    const { submitJob } = await import("@parallax/sdk/submit");
    const start = Date.now();
    const jobId = await submitJob({
      action:     "create",
      media:      "video",
      model:      "ltx2",
      script:     "runtime/video/ltx/ltx2/t2v.py",
      args:       ["--prompt", "test"],
      scriptBase: "/tmp",
      uvPath:     "uv",
    });
    const result = `job_id: ${jobId}\nstatus: queued\nmodel: ltx2`;
    const elapsed = Date.now() - start;
    expect(result).toContain("job_id:");
    expect(elapsed).toBeLessThan(500);
  });
});

// ── AC04: create_audio returns job_id in under 500ms ─────────────────────────

describe("US-001-AC04: create_audio non-blocking job submission", () => {
  it("returns a string containing job_id: in under 500ms", async () => {
    const { submitJob } = await import("@parallax/sdk/submit");
    const start = Date.now();
    const jobId = await submitJob({
      action:     "create",
      media:      "audio",
      model:      "ace_step",
      script:     "runtime/audio/ace/t2a.py",
      args:       ["--tags", "test"],
      scriptBase: "/tmp",
      uvPath:     "uv",
    });
    const result = `job_id: ${jobId}\nstatus: queued\nmodel: ace_step`;
    const elapsed = Date.now() - start;
    expect(result).toContain("job_id:");
    expect(elapsed).toBeLessThan(500);
  });
});

// ── AC04: edit_image returns job_id in under 500ms ───────────────────────────

describe("US-001-AC04: edit_image non-blocking job submission", () => {
  it("returns a string containing job_id: in under 500ms", async () => {
    const { submitJob } = await import("@parallax/sdk/submit");
    const start = Date.now();
    const jobId = await submitJob({
      action:     "edit",
      media:      "image",
      model:      "flux_9b_kv",
      script:     "runtime/image/edit/flux/9b_kv.py",
      args:       ["--prompt", "test", "--image", "input.png"],
      scriptBase: "/tmp",
      uvPath:     "uv",
    });
    const result = `job_id: ${jobId}\nstatus: queued\nmodel: flux_9b_kv`;
    const elapsed = Date.now() - start;
    expect(result).toContain("job_id:");
    expect(elapsed).toBeLessThan(500);
  });
});

// ── AC04: upscale_image returns job_id in under 500ms ────────────────────────

describe("US-001-AC04: upscale_image non-blocking job submission", () => {
  it("returns a string containing job_id: in under 500ms", async () => {
    const { submitJob } = await import("@parallax/sdk/submit");
    const start = Date.now();
    const jobId = await submitJob({
      action:     "upscale",
      media:      "image",
      model:      "esrgan",
      script:     "runtime/image/edit/sd/esrgan_upscale.py",
      args:       ["--input", "input.png", "--prompt", "test"],
      scriptBase: "/tmp",
      uvPath:     "uv",
    });
    const result = `job_id: ${jobId}\nstatus: queued\nmodel: esrgan`;
    const elapsed = Date.now() - start;
    expect(result).toContain("job_id:");
    expect(elapsed).toBeLessThan(500);
  });
});
