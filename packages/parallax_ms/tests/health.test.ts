// Tests for US-006: Health endpoint enrichment.
// Strategy:
//   - Structural tests: check source for getQueueStats import and queue field.
//   - Functional tests: use app.handle() with mocked getQueueStats to verify
//     response shape and HTTP 200 status.

import { describe, it, expect, mock } from "bun:test";
import { readFileSync } from "fs";
import { join } from "path";

import type { QueueStats } from "@parallax/sdk";

// ── Mock @parallax/sdk BEFORE importing the app ────────────────────────────────
const mockGetQueueStats = mock(async (): Promise<QueueStats> => ({
  waiting: 2,
  active: 1,
  completed: 10,
  failed: 3,
}));

mock.module("@parallax/sdk", () => ({
  submitJob: mock(async () => "mock-id"),
  getJobStatus: mock(async () => null),
  listJobs: mock(async () => ({ jobs: [], counts: { waiting: 0, active: 0, completed: 0, failed: 0 } })),
  cancelJob: mock(async () => null),
  getQueueStats: mockGetQueueStats,
}));

import { app } from "../src/index";

const SRC = readFileSync(join(import.meta.dir, "../src/index.ts"), "utf-8");

describe("US-006 — Health endpoint enrichment", () => {
  // ── Structural tests ──────────────────────────────────────────────────────────
  it("US-006-AC03 (structural): imports getQueueStats from @parallax/sdk", () => {
    expect(SRC).toContain("getQueueStats");
  });

  it("US-006-AC01 (structural): /health route calls getQueueStats and returns queue field", () => {
    expect(SRC).toContain("getQueueStats");
    expect(SRC).toContain("queue");
  });

  // ── Functional tests ──────────────────────────────────────────────────────────
  it("US-006-AC02: GET /health returns HTTP 200", async () => {
    const res = await app.handle(new Request("http://localhost/health"));
    expect(res.status).toBe(200);
  });

  it("US-006-AC01: GET /health returns { status: 'ok', queue: { waiting, active, completed, failed } }", async () => {
    const res = await app.handle(new Request("http://localhost/health"));
    const body = await res.json();

    expect(body).toMatchObject({
      status: "ok",
      queue: {
        waiting: 2,
        active: 1,
        completed: 10,
        failed: 3,
      },
    });
  });

  it("US-006-AC01: queue object contains all four counters as numbers", async () => {
    const res = await app.handle(new Request("http://localhost/health"));
    const body = await res.json();

    expect(typeof body.queue.waiting).toBe("number");
    expect(typeof body.queue.active).toBe("number");
    expect(typeof body.queue.completed).toBe("number");
    expect(typeof body.queue.failed).toBe("number");
  });

  it("US-006-AC01: queue reflects values returned by getQueueStats", async () => {
    mockGetQueueStats.mockImplementationOnce(async () => ({
      waiting: 0,
      active: 5,
      completed: 42,
      failed: 1,
    }));

    const res = await app.handle(new Request("http://localhost/health"));
    const body = await res.json();

    expect(body.queue).toEqual({ waiting: 0, active: 5, completed: 42, failed: 1 });
  });
});
