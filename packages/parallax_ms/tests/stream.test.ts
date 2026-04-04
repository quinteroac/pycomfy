// Tests for US-003: Stream job progress via SSE.
// Strategy:
//   - AC05: 404 returned synchronously before stream opens when job does not exist.
//   - AC01: Content-Type is text/event-stream for a known job.
//   - AC02: progress events emitted while job is active/waiting.
//   - AC03: completed event emitted and stream closed on job completion.
//   - AC04: failed event emitted and stream closed on job failure.
//   - AC06: @elysiajs/stream listed in package.json dependencies.

import { describe, it, expect, mock, beforeEach } from "bun:test";
import { readFileSync } from "fs";
import { join } from "path";

// ── Mock @parallax/sdk BEFORE importing the app ───────────────────────────────
const mockGetJobStatus = mock(async (_id: string) => null as any);

mock.module("@parallax/sdk", () => ({
  submitJob: mock(async () => "mock-job-id-003"),
  getJobStatus: mockGetJobStatus,
}));

import { app } from "../src/index";

// ── Helpers ───────────────────────────────────────────────────────────────────
const BASE = "http://localhost";

const waitingJob = {
  id: "test-job",
  status: "waiting" as const,
  progress: 0,
  output: null,
  error: null,
  createdAt: 1700000000000,
  startedAt: null,
  finishedAt: null,
};

const activeJob = {
  id: "test-job",
  status: "active" as const,
  progress: 42,
  output: null,
  error: null,
  createdAt: 1700000000000,
  startedAt: 1700000001000,
  finishedAt: null,
};

const completedJob = {
  id: "test-job",
  status: "completed" as const,
  progress: 100,
  output: "/output/result.png",
  error: null,
  createdAt: 1700000000000,
  startedAt: 1700000001000,
  finishedAt: 1700000060000,
};

const failedJob = {
  id: "test-job",
  status: "failed" as const,
  progress: 0,
  output: null,
  error: "CUDA out of memory",
  createdAt: 1700000000000,
  startedAt: 1700000001000,
  finishedAt: 1700000010000,
};

beforeEach(() => {
  mockGetJobStatus.mockReset();
});

// ── Tests ─────────────────────────────────────────────────────────────────────
describe("GET /jobs/:id/stream", () => {
  // AC05: 404 before opening stream when job does not exist
  it("returns 404 JSON when job does not exist", async () => {
    mockGetJobStatus.mockImplementation(async () => null);

    const resp = await app.handle(
      new Request(`${BASE}/jobs/nonexistent/stream`),
    );
    expect(resp.status).toBe(404);
    const body = await resp.json();
    expect(body).toHaveProperty("error");
  });

  // AC01: response is text/event-stream
  it("returns text/event-stream content type for a known job", async () => {
    // existence check returns active, first poll returns completed → closes stream
    let callCount = 0;
    mockGetJobStatus.mockImplementation(async () => {
      return callCount++ === 0 ? activeJob : completedJob;
    });

    const resp = await app.handle(
      new Request(`${BASE}/jobs/test-job/stream`),
    );
    expect(resp.status).toBe(200);
    expect(resp.headers.get("content-type")).toContain("text/event-stream");
    // drain stream to avoid resource leak
    await resp.text();
  }, { timeout: 2000 });

  // AC03: emits completed event then closes stream
  it("emits completed event with output path when job finishes", async () => {
    let callCount = 0;
    mockGetJobStatus.mockImplementation(async () => {
      return callCount++ === 0 ? activeJob : completedJob;
    });

    const resp = await app.handle(
      new Request(`${BASE}/jobs/test-job/stream`),
    );
    const text = await resp.text(); // waits for stream to close

    expect(text).toContain("event: completed");
    expect(text).toContain('"output"');
    expect(text).toContain("/output/result.png");
  }, { timeout: 2000 });

  // AC04: emits failed event then closes stream
  it("emits failed event with error reason when job fails", async () => {
    let callCount = 0;
    mockGetJobStatus.mockImplementation(async () => {
      return callCount++ === 0 ? activeJob : failedJob;
    });

    const resp = await app.handle(
      new Request(`${BASE}/jobs/test-job/stream`),
    );
    const text = await resp.text();

    expect(text).toContain("event: failed");
    expect(text).toContain('"error"');
    expect(text).toContain("CUDA out of memory");
  }, { timeout: 2000 });

  // AC02: emits progress event while job is active/waiting
  it("emits progress events while job is active", async () => {
    let callCount = 0;
    mockGetJobStatus.mockImplementation(async () => {
      callCount++;
      if (callCount === 1) return activeJob;  // existence check
      if (callCount === 2) return activeJob;  // first poll → progress
      return completedJob;                    // second poll → completed → close
    });

    const resp = await app.handle(
      new Request(`${BASE}/jobs/test-job/stream`),
    );
    const text = await resp.text();

    expect(text).toContain("event: progress");
    expect(text).toContain('"pct"');
    expect(text).toContain('"step"');
    // stream must also close with completed event
    expect(text).toContain("event: completed");
  }, { timeout: 3000 });

  // AC06: @elysiajs/stream is listed as a dependency
  it("lists @elysiajs/stream as a dependency in package.json", () => {
    const pkgPath = join(import.meta.dir, "..", "package.json");
    const pkg = JSON.parse(readFileSync(pkgPath, "utf-8"));
    expect(pkg.dependencies).toHaveProperty("@elysiajs/stream");
  });
});
