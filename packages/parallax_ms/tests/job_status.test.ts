// Tests for US-002: Get job status by ID.
// Strategy:
//   - Structural tests: verify GET /jobs/:id route is defined in source.
//   - Functional tests: use app.handle() with mocked getJobStatus to verify
//     response shapes, field types, and 404 behaviour.

import { describe, it, expect, mock } from "bun:test";
import { readFileSync } from "fs";
import { join } from "path";

// ── Mock @parallax/sdk BEFORE importing the app ───────────────────────────────
mock.module("@parallax/sdk", () => ({
  submitJob: mock(async () => "mock-job-id-002"),
  getJobStatus: mock(async (id: string) => {
    if (id === "existing-job") {
      return {
        id: "existing-job",
        status: "completed",
        progress: 100,
        output: "/tmp/output.png",
        error: null,
        createdAt: 1700000000000,
        startedAt: 1700000001000,
        finishedAt: 1700000060000,
      };
    }
    if (id === "active-job") {
      return {
        id: "active-job",
        status: "active",
        progress: 50,
        output: null,
        error: null,
        createdAt: 1700000000000,
        startedAt: 1700000001000,
        finishedAt: null,
      };
    }
    if (id === "failed-job") {
      return {
        id: "failed-job",
        status: "failed",
        progress: 0,
        output: null,
        error: "CUDA out of memory",
        createdAt: 1700000000000,
        startedAt: 1700000001000,
        finishedAt: 1700000010000,
      };
    }
    if (id === "waiting-job") {
      return {
        id: "waiting-job",
        status: "waiting",
        progress: 0,
        output: null,
        error: null,
        createdAt: 1700000000000,
        startedAt: null,
        finishedAt: null,
      };
    }
    return null;
  }),
}));

import { app } from "../src/index";

const SRC = readFileSync(join(import.meta.dir, "../src/index.ts"), "utf-8");

// Helper: GET from app
async function get(path: string) {
  return app.handle(new Request(`http://localhost${path}`));
}

// ── Structural: route defined in source ───────────────────────────────────────

describe("US-002 structural: GET /jobs/:id route", () => {
  it("defines GET /jobs/:id route", () => {
    expect(SRC).toContain('"/jobs/:id"');
  });

  it("calls getJobStatus from @parallax/sdk", () => {
    expect(SRC).toContain("getJobStatus");
  });

  it("returns 404 with error message for missing job", () => {
    expect(SRC).toContain("Job not found");
  });
});

// ── Functional: AC01 — response shape for existing job ───────────────────────

describe("US-002-AC01 functional: job status response shape", () => {
  it("returns 200 for an existing job", async () => {
    const res = await get("/jobs/existing-job");
    expect(res.status).toBe(200);
  });

  it("response includes id field", async () => {
    const res = await get("/jobs/existing-job");
    const json = await res.json() as Record<string, unknown>;
    expect(json.id).toBe("existing-job");
  });

  it("response includes status field (completed)", async () => {
    const res = await get("/jobs/existing-job");
    const json = await res.json() as Record<string, unknown>;
    expect(json.status).toBe("completed");
  });

  it("response includes progress as a number", async () => {
    const res = await get("/jobs/existing-job");
    const json = await res.json() as Record<string, unknown>;
    expect(typeof json.progress).toBe("number");
    expect(json.progress).toBe(100);
  });

  it("response includes output field (string)", async () => {
    const res = await get("/jobs/existing-job");
    const json = await res.json() as Record<string, unknown>;
    expect(json.output).toBe("/tmp/output.png");
  });

  it("response includes error field (null for completed)", async () => {
    const res = await get("/jobs/existing-job");
    const json = await res.json() as Record<string, unknown>;
    expect(json.error).toBeNull();
  });

  it("response includes createdAt as epoch ms number", async () => {
    const res = await get("/jobs/existing-job");
    const json = await res.json() as Record<string, unknown>;
    expect(typeof json.createdAt).toBe("number");
    expect(json.createdAt).toBe(1700000000000);
  });

  it("response includes startedAt as epoch ms number", async () => {
    const res = await get("/jobs/existing-job");
    const json = await res.json() as Record<string, unknown>;
    expect(typeof json.startedAt).toBe("number");
    expect(json.startedAt).toBe(1700000001000);
  });

  it("response includes finishedAt as epoch ms number", async () => {
    const res = await get("/jobs/existing-job");
    const json = await res.json() as Record<string, unknown>;
    expect(typeof json.finishedAt).toBe("number");
    expect(json.finishedAt).toBe(1700000060000);
  });
});

// ── Functional: AC01 — status variants ───────────────────────────────────────

describe("US-002-AC01 functional: all valid status values", () => {
  it("returns active status for active job", async () => {
    const res = await get("/jobs/active-job");
    const json = await res.json() as Record<string, unknown>;
    expect(json.status).toBe("active");
    expect(json.progress).toBe(50);
    expect(json.finishedAt).toBeNull();
  });

  it("returns failed status with error string", async () => {
    const res = await get("/jobs/failed-job");
    const json = await res.json() as Record<string, unknown>;
    expect(json.status).toBe("failed");
    expect(typeof json.error).toBe("string");
    expect(json.error).toBe("CUDA out of memory");
  });

  it("returns waiting status with null startedAt and finishedAt", async () => {
    const res = await get("/jobs/waiting-job");
    const json = await res.json() as Record<string, unknown>;
    expect(json.status).toBe("waiting");
    expect(json.startedAt).toBeNull();
    expect(json.finishedAt).toBeNull();
  });
});

// ── Functional: AC02 — 404 for missing job ───────────────────────────────────

describe("US-002-AC02 functional: 404 for unknown job ID", () => {
  it("returns 404 status for non-existent job", async () => {
    const res = await get("/jobs/nonexistent-id");
    expect(res.status).toBe(404);
  });

  it("404 response includes { error: 'Job not found' }", async () => {
    const res = await get("/jobs/nonexistent-id");
    const json = await res.json() as Record<string, unknown>;
    expect(json.error).toBe("Job not found");
  });
});
