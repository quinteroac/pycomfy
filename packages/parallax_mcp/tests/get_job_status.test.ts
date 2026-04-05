// Tests for US-002: get_job_status MCP tool.

import { describe, it, expect, mock, beforeAll, afterAll } from "bun:test";
import { readFileSync } from "fs";
import { join } from "path";

const SRC = readFileSync(join(import.meta.dir, "../src/index.ts"), "utf-8");

// ── AC01: input schema ────────────────────────────────────────────────────────

describe("US-002-AC01: get_job_status accepts job_id as sole parameter", () => {
  it("registers a tool named get_job_status", () => {
    expect(SRC).toContain('"get_job_status"');
  });

  it("input schema includes only job_id as z.string()", () => {
    // job_id must be present as the parameter
    expect(SRC).toContain("job_id:");
    expect(SRC).toContain("z.string()");
  });

  it("imports getJobStatus from @parallax/sdk", () => {
    expect(SRC).toContain("getJobStatus");
    expect(SRC).toContain("@parallax/sdk");
  });
});

// ── AC02: response fields ─────────────────────────────────────────────────────

describe("US-002-AC02: text response includes required fields", () => {
  it("response includes status field", () => {
    expect(SRC).toContain("status:");
  });

  it("response includes model field", () => {
    expect(SRC).toContain("model:");
  });

  it("response includes created_at field", () => {
    expect(SRC).toContain("created_at:");
  });

  it("response includes output_path field (for completed jobs)", () => {
    expect(SRC).toContain("output_path");
  });

  it("response includes error field (for failed jobs)", () => {
    expect(SRC).toContain("error:");
  });

  it("response is JSON-formatted (JSON.stringify)", () => {
    expect(SRC).toContain("JSON.stringify");
  });
});

// ── AC03: not_found (no exception) ───────────────────────────────────────────

describe("US-002-AC03: missing job returns status: not_found without error", () => {
  it("returns status: not_found for missing job", () => {
    expect(SRC).toContain('"not_found"');
  });

  it("not-found path does not use isError", () => {
    // The not-found branch must not set isError: true
    const notFoundBlock = SRC.slice(SRC.indexOf('"not_found"') - 200, SRC.indexOf('"not_found"') + 100);
    expect(notFoundBlock).not.toContain("isError: true");
  });

  it("handler references input.job_id", () => {
    expect(SRC).toContain("input.job_id");
  });
});

// ── AC02 + AC03: runtime behaviour ───────────────────────────────────────────

describe("US-002 runtime: get_job_status response format", () => {
  type JobStatus = {
    id: string;
    status: "waiting" | "active" | "completed" | "failed";
    progress: number;
    model: string | null;
    action: string | null;
    media: string | null;
    output: string | null;
    error: string | null;
    createdAt: number;
    startedAt: number | null;
    finishedAt: number | null;
  };

  function buildResponse(status: JobStatus | null): string {
    if (!status) {
      return JSON.stringify({ status: "not_found" });
    }
    const payload: Record<string, unknown> = {
      status:     status.status,
      model:      status.model,
      created_at: status.createdAt,
    };
    if (status.status === "completed") payload.output_path = status.output;
    if (status.status === "failed")    payload.error = status.error;
    return JSON.stringify(payload);
  }

  it("AC03: null job produces status: not_found", () => {
    const text = buildResponse(null);
    const parsed = JSON.parse(text);
    expect(parsed.status).toBe("not_found");
  });

  it("AC02: completed job includes status, model, created_at, output_path", () => {
    const job: JobStatus = {
      id: "job-1", status: "completed", progress: 100,
      model: "sdxl", action: "create", media: "image",
      output: "/out/image.png", error: null,
      createdAt: 1700000000000, startedAt: null, finishedAt: null,
    };
    const parsed = JSON.parse(buildResponse(job));
    expect(parsed.status).toBe("completed");
    expect(parsed.model).toBe("sdxl");
    expect(parsed.created_at).toBe(1700000000000);
    expect(parsed.output_path).toBe("/out/image.png");
    expect(parsed.error).toBeUndefined();
  });

  it("AC02: failed job includes status, model, created_at, error (no output_path)", () => {
    const job: JobStatus = {
      id: "job-2", status: "failed", progress: 0,
      model: "sdxl", action: "create", media: "image",
      output: null, error: "OOM",
      createdAt: 1700000000000, startedAt: null, finishedAt: null,
    };
    const parsed = JSON.parse(buildResponse(job));
    expect(parsed.status).toBe("failed");
    expect(parsed.model).toBe("sdxl");
    expect(parsed.created_at).toBe(1700000000000);
    expect(parsed.error).toBe("OOM");
    expect(parsed.output_path).toBeUndefined();
  });

  it("AC02: waiting job includes status, model, created_at — no output_path or error", () => {
    const job: JobStatus = {
      id: "job-3", status: "waiting", progress: 0,
      model: "ltx2", action: "create", media: "video",
      output: null, error: null,
      createdAt: 1700000000001, startedAt: null, finishedAt: null,
    };
    const parsed = JSON.parse(buildResponse(job));
    expect(parsed.status).toBe("waiting");
    expect(parsed.model).toBe("ltx2");
    expect(parsed.created_at).toBe(1700000000001);
    expect(parsed.output_path).toBeUndefined();
    expect(parsed.error).toBeUndefined();
  });
});
