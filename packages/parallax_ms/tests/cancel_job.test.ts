// Tests for US-005: Cancel a job.
// Strategy:
//   - Structural tests: verify DELETE /jobs/:id route is defined in source.
//   - Functional tests: use app.handle() with mocked cancelJob to verify
//     response shapes, 404, and 409 behaviour.

import { describe, it, expect, mock } from "bun:test";
import { readFileSync } from "fs";
import { join } from "path";

// ── Mock @parallax/sdk BEFORE importing the app ───────────────────────────────
mock.module("@parallax/sdk", () => ({
  submitJob:    mock(async () => "mock-job-id-005"),
  getJobStatus: mock(async () => null),
  listJobs:     mock(async () => ({ jobs: [], counts: { waiting: 0, active: 0, completed: 0, failed: 0 } })),
  cancelJob:    mock(async (id: string) => {
    if (id === "active-job")    return true;
    if (id === "waiting-job")   return true;
    if (id === "done-job")      return "terminal";
    if (id === "failed-job")    return "terminal";
    return null; // not found
  }),
}));

const { app } = await import("../src/index");

// ── Structural tests ──────────────────────────────────────────────────────────
describe("US-005 structural", () => {
  it("defines DELETE /jobs/:id route in source", () => {
    const src = readFileSync(
      join(import.meta.dir, "../src/index.ts"),
      "utf8",
    );
    expect(src).toContain('.delete("/jobs/:id"');
  });

  it("imports cancelJob from @parallax/sdk", () => {
    const src = readFileSync(
      join(import.meta.dir, "../src/index.ts"),
      "utf8",
    );
    expect(src).toContain("cancelJob");
  });
});

// ── Functional tests ──────────────────────────────────────────────────────────
describe("US-005-AC01 — cancel returns { cancelled: true }", () => {
  it("returns 200 with { cancelled: true } for an active job", async () => {
    const resp = await app.handle(
      new Request("http://localhost/jobs/active-job", { method: "DELETE" }),
    );
    expect(resp.status).toBe(200);
    const body = await resp.json();
    expect(body).toEqual({ cancelled: true });
  });

  it("returns 200 with { cancelled: true } for a waiting job", async () => {
    const resp = await app.handle(
      new Request("http://localhost/jobs/waiting-job", { method: "DELETE" }),
    );
    expect(resp.status).toBe(200);
    const body = await resp.json();
    expect(body).toEqual({ cancelled: true });
  });
});

describe("US-005-AC02 — 404 when job does not exist", () => {
  it("returns 404 with error field for unknown job ID", async () => {
    const resp = await app.handle(
      new Request("http://localhost/jobs/nonexistent-job-xyz", { method: "DELETE" }),
    );
    expect(resp.status).toBe(404);
    const body = await resp.json();
    expect(body).toHaveProperty("error");
  });
});

describe("US-005-AC03 — 409 when job is already in terminal state", () => {
  it("returns 409 with { error: 'Job already completed' } for a completed job", async () => {
    const resp = await app.handle(
      new Request("http://localhost/jobs/done-job", { method: "DELETE" }),
    );
    expect(resp.status).toBe(409);
    const body = await resp.json();
    expect(body).toEqual({ error: "Job already completed" });
  });

  it("returns 409 with { error: 'Job already completed' } for a failed job", async () => {
    const resp = await app.handle(
      new Request("http://localhost/jobs/failed-job", { method: "DELETE" }),
    );
    expect(resp.status).toBe(409);
    const body = await resp.json();
    expect(body).toEqual({ error: "Job already completed" });
  });
});
