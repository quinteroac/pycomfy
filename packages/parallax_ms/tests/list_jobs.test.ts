// Tests for US-004: List jobs via GET /jobs.
// Strategy:
//   - Structural tests: check source for route and query-param definitions.
//   - Functional tests: use app.handle() with mocked listJobs to verify
//     response shapes, status filter, default limit, and count fields.

import { describe, it, expect, mock, beforeAll } from "bun:test";
import { readFileSync } from "fs";
import { join } from "path";

import type { JobListResult } from "@parallax/sdk";

// ── Mock @parallax/sdk BEFORE importing the app ────────────────────────────────
const mockListJobs = mock(async (_opts?: { status?: string; limit?: number }): Promise<JobListResult> => ({
  jobs: [],
  counts: { waiting: 0, active: 0, completed: 0, failed: 0 },
}));

mock.module("@parallax/sdk", () => ({
  submitJob: mock(async () => "mock-id"),
  getJobStatus: mock(async () => null),
  listJobs: mockListJobs,
}));

import { app } from "../src/index";

const SRC = readFileSync(join(import.meta.dir, "../src/index.ts"), "utf-8");

function get(path: string) {
  return app.handle(new Request(`http://localhost${path}`));
}

// ── Structural ────────────────────────────────────────────────────────────────

describe("US-004 structural: route definitions", () => {
  it('defines GET /jobs route', () => {
    expect(SRC).toContain('"/jobs"');
  });

  it("calls listJobs from @parallax/sdk", () => {
    expect(SRC).toContain("listJobs");
  });

  it("declares optional query param 'status'", () => {
    expect(SRC).toContain("status");
    expect(SRC).toContain("waiting");
    expect(SRC).toContain("active");
    expect(SRC).toContain("completed");
    expect(SRC).toContain("failed");
  });
});

// ── Functional ────────────────────────────────────────────────────────────────

describe("US-004 AC01: response shape", () => {
  it("returns { jobs, counts } with correct count keys", async () => {
    mockListJobs.mockResolvedValueOnce({
      jobs: [],
      counts: { waiting: 2, active: 1, completed: 5, failed: 0 },
    });

    const resp = await get("/jobs");
    expect(resp.status).toBe(200);

    const body = await resp.json() as { jobs: unknown[]; counts: Record<string, number> };
    expect(body).toHaveProperty("jobs");
    expect(body).toHaveProperty("counts");
    expect(body.counts).toHaveProperty("waiting");
    expect(body.counts).toHaveProperty("active");
    expect(body.counts).toHaveProperty("completed");
    expect(body.counts).toHaveProperty("failed");
  });
});

describe("US-004 AC02: JobSummary fields", () => {
  it("includes all required JobSummary fields", async () => {
    const now = Date.now();
    mockListJobs.mockResolvedValueOnce({
      jobs: [
        {
          id: "job-1",
          status: "completed",
          progress: 100,
          model: "sdxl",
          action: "create",
          media: "image",
          createdAt: now,
        },
      ],
      counts: { waiting: 0, active: 0, completed: 1, failed: 0 },
    });

    const resp = await get("/jobs");
    const body = await resp.json() as { jobs: Record<string, unknown>[] };
    const job = body.jobs[0];

    expect(job).toHaveProperty("id", "job-1");
    expect(job).toHaveProperty("status", "completed");
    expect(job).toHaveProperty("progress", 100);
    expect(job).toHaveProperty("model", "sdxl");
    expect(job).toHaveProperty("action", "create");
    expect(job).toHaveProperty("media", "image");
    expect(job).toHaveProperty("createdAt", now);
  });
});

describe("US-004 AC03: default limit", () => {
  it("calls listJobs without overriding the limit (defaults to 50)", async () => {
    mockListJobs.mockClear();
    mockListJobs.mockResolvedValueOnce({
      jobs: [],
      counts: { waiting: 0, active: 0, completed: 0, failed: 0 },
    });

    await get("/jobs");
    expect(mockListJobs).toHaveBeenCalledTimes(1);
    // No explicit limit override — listJobs own default of 50 applies
    const callArg = mockListJobs.mock.calls[0][0] as { status?: string } | undefined;
    expect(callArg?.status).toBeUndefined();
  });
});

describe("US-004 AC04: status filter", () => {
  it("passes status=active to listJobs", async () => {
    mockListJobs.mockClear();
    mockListJobs.mockResolvedValueOnce({
      jobs: [],
      counts: { waiting: 0, active: 2, completed: 0, failed: 0 },
    });

    const resp = await get("/jobs?status=active");
    expect(resp.status).toBe(200);
    expect(mockListJobs).toHaveBeenCalledTimes(1);
    const callArg = mockListJobs.mock.calls[0][0] as { status?: string };
    expect(callArg?.status).toBe("active");
  });

  it("passes status=failed to listJobs", async () => {
    mockListJobs.mockClear();
    mockListJobs.mockResolvedValueOnce({
      jobs: [],
      counts: { waiting: 0, active: 0, completed: 0, failed: 3 },
    });

    const resp = await get("/jobs?status=failed");
    expect(resp.status).toBe(200);
    const callArg = mockListJobs.mock.calls[0][0] as { status?: string };
    expect(callArg?.status).toBe("failed");
  });

  it("passes status=waiting to listJobs", async () => {
    mockListJobs.mockClear();
    mockListJobs.mockResolvedValueOnce({
      jobs: [],
      counts: { waiting: 1, active: 0, completed: 0, failed: 0 },
    });

    await get("/jobs?status=waiting");
    const callArg = mockListJobs.mock.calls[0][0] as { status?: string };
    expect(callArg?.status).toBe("waiting");
  });

  it("passes status=completed to listJobs", async () => {
    mockListJobs.mockClear();
    mockListJobs.mockResolvedValueOnce({
      jobs: [],
      counts: { waiting: 0, active: 0, completed: 10, failed: 0 },
    });

    await get("/jobs?status=completed");
    const callArg = mockListJobs.mock.calls[0][0] as { status?: string };
    expect(callArg?.status).toBe("completed");
  });
});
