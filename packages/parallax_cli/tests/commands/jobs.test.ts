// Tests for US-002: parallax jobs list
// Strategy: test pure helper functions (colorStatus, formatStarted, buildRows, formatJobsTable).
// Integration of listJobs and the Commander action handler is validated via bun typecheck (AC05).

import { describe, it, expect } from "bun:test";
import {
  colorStatus,
  formatStarted,
  buildRows,
  formatJobsTable,
  EMPTY_MESSAGE,
} from "../../src/commands/jobs";
import type { JobSummary } from "@parallax/sdk/list";

// --- helpers ---

function makeJob(overrides: Partial<JobSummary> = {}): JobSummary {
  return {
    id: "1",
    status: "completed",
    progress: 100,
    model: "sdxl",
    action: "create",
    media: "image",
    createdAt: Date.now() - 5000,
    ...overrides,
  };
}

// Strip ANSI escape sequences for readable assertions.
function strip(s: string): string {
  return s.replace(/\x1b\[[0-9;]*m/g, "");
}

// US-002-AC02: status color coding
describe("US-002-AC02: colorStatus", () => {
  it("waiting is dimmed (ANSI dim code \\x1b[2m)", () => {
    expect(colorStatus("waiting")).toContain("\x1b[2m");
  });

  it("active contains cyan code \\x1b[36m", () => {
    expect(colorStatus("active")).toContain("\x1b[36m");
  });

  it("completed contains green code \\x1b[32m", () => {
    expect(colorStatus("completed")).toContain("\x1b[32m");
  });

  it("failed contains red code \\x1b[31m", () => {
    expect(colorStatus("failed")).toContain("\x1b[31m");
  });

  it("each status includes the status text", () => {
    for (const s of ["waiting", "active", "completed", "failed"] as const) {
      expect(strip(colorStatus(s))).toBe(s);
    }
  });
});

// US-002-AC01: table columns
describe("US-002-AC01: table columns", () => {
  const jobs = [makeJob()];
  const table = formatJobsTable(jobs);

  it("includes ID header", () => {
    expect(table).toContain("ID");
  });

  it("includes Status header", () => {
    expect(table).toContain("Status");
  });

  it("includes Action header", () => {
    expect(table).toContain("Action");
  });

  it("includes Model header", () => {
    expect(table).toContain("Model");
  });

  it("includes Progress header", () => {
    expect(table).toContain("Progress");
  });

  it("includes Started header", () => {
    expect(table).toContain("Started");
  });

  it("includes Duration header", () => {
    expect(table).toContain("Duration");
  });

  it("renders job data in the output", () => {
    expect(strip(table)).toContain("sdxl");
    expect(strip(table)).toContain("create");
    expect(strip(table)).toContain("100%");
  });
});

// US-002-AC04: empty state message
describe("US-002-AC04: empty jobs", () => {
  it("returns the exact empty message when jobs array is empty", () => {
    expect(formatJobsTable([])).toBe(EMPTY_MESSAGE);
  });

  it("empty message text matches spec", () => {
    expect(EMPTY_MESSAGE).toBe(
      "No jobs found. Run a command with --async to submit one.",
    );
  });
});

// US-002-AC03: at most 20 most-recent jobs
describe("US-002-AC03: at most 20 jobs", () => {
  // The limit is enforced by listJobs({ limit: 20 }) in the SDK.
  // buildRows must handle any number of jobs passed to it.
  it("buildRows returns one row per job", () => {
    const jobs = Array.from({ length: 5 }, (_, i) =>
      makeJob({ id: String(i) }),
    );
    const rows = buildRows(jobs);
    expect(rows).toHaveLength(5);
  });

  it("formatJobsTable renders all provided rows", () => {
    const jobs = Array.from({ length: 3 }, (_, i) =>
      makeJob({ id: `job-${i}` }),
    );
    const table = formatJobsTable(jobs);
    for (let i = 0; i < 3; i++) {
      expect(table).toContain(`job-${i}`);
    }
  });
});

// formatStarted basic sanity
describe("formatStarted", () => {
  it("returns seconds-ago for recent timestamps", () => {
    const result = formatStarted(Date.now() - 30_000);
    expect(result).toMatch(/^\d+s ago$/);
  });

  it("returns minutes-ago for timestamps a few minutes old", () => {
    const result = formatStarted(Date.now() - 5 * 60_000);
    expect(result).toMatch(/^\d+m ago$/);
  });

  it("returns hours-ago for timestamps several hours old", () => {
    const result = formatStarted(Date.now() - 3 * 60 * 60_000);
    expect(result).toMatch(/^\d+h ago$/);
  });
});
