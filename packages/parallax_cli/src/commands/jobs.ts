// jobs command: `parallax jobs list`
// Displays a table of recent jobs with color-coded status.

import { Command } from "commander";
import { listJobs } from "@parallax/sdk/list";
import type { JobSummary, JobStatusValue } from "@parallax/sdk/list";

export const EMPTY_MESSAGE =
  "No jobs found. Run a command with --async to submit one.";

// ANSI color codes — avoid an extra dependency since picocolors is only a transitive dep.
const RESET = "\x1b[0m";
const DIM = "\x1b[2m";
const CYAN = "\x1b[36m";
const GREEN = "\x1b[32m";
const RED = "\x1b[31m";

export function colorStatus(status: JobStatusValue): string {
  switch (status) {
    case "waiting":
      return `${DIM}${status}${RESET}`;
    case "active":
      return `${CYAN}${status}${RESET}`;
    case "completed":
      return `${GREEN}${status}${RESET}`;
    case "failed":
      return `${RED}${status}${RESET}`;
  }
}

// Format a Unix millisecond timestamp as a human-readable relative time.
export function formatStarted(createdAt: number): string {
  const diffMs = Date.now() - createdAt;
  const diffSec = Math.floor(diffMs / 1000);
  if (diffSec < 60) return `${diffSec}s ago`;
  const diffMin = Math.floor(diffSec / 60);
  if (diffMin < 60) return `${diffMin}m ago`;
  const diffHr = Math.floor(diffMin / 60);
  if (diffHr < 24) return `${diffHr}h ago`;
  return `${Math.floor(diffHr / 24)}d ago`;
}

export interface TableRow {
  id: string;
  status: string;
  action: string;
  model: string;
  progress: string;
  started: string;
  duration: string;
}

const HEADERS: Array<keyof TableRow> = [
  "id",
  "status",
  "action",
  "model",
  "progress",
  "started",
  "duration",
];

const HEADER_LABELS: Record<keyof TableRow, string> = {
  id: "ID",
  status: "Status",
  action: "Action",
  model: "Model",
  progress: "Progress",
  started: "Started",
  duration: "Duration",
};

// Strip ANSI escape sequences to measure visible string length.
function visibleLength(s: string): number {
  return s.replace(/\x1b\[[0-9;]*m/g, "").length;
}

export function buildRows(jobs: JobSummary[]): TableRow[] {
  return jobs.map((job) => ({
    id: job.id,
    status: colorStatus(job.status),
    action: job.action || "—",
    model: job.model || "—",
    progress: `${job.progress}%`,
    started: formatStarted(job.createdAt),
    duration: "—",
  }));
}

// Build and return the formatted table string (no ANSI in column widths calculation).
export function formatJobsTable(jobs: JobSummary[]): string {
  if (jobs.length === 0) return EMPTY_MESSAGE;

  const rows = buildRows(jobs);

  // Column widths based on visible (ANSI-stripped) content.
  const widths: Record<keyof TableRow, number> = {} as Record<
    keyof TableRow,
    number
  >;
  for (const key of HEADERS) {
    widths[key] = HEADER_LABELS[key].length;
    for (const row of rows) {
      widths[key] = Math.max(widths[key], visibleLength(row[key]));
    }
  }

  const padCol = (s: string, width: number) =>
    s + " ".repeat(Math.max(0, width - visibleLength(s)));

  const headerLine = HEADERS.map((k) =>
    padCol(HEADER_LABELS[k], widths[k]),
  ).join("  ");
  const separator = HEADERS.map((k) => "─".repeat(widths[k])).join("  ");
  const dataLines = rows.map((row) =>
    HEADERS.map((k) => padCol(row[k], widths[k])).join("  "),
  );

  return [headerLine, separator, ...dataLines].join("\n");
}

export function registerJobs(program: Command): void {
  const jobs = program
    .command("jobs")
    .description("Manage background jobs");

  jobs
    .command("list")
    .description("Show recent jobs (newest first, max 20)")
    .action(async () => {
      const result = await listJobs({ limit: 20 });
      console.log(formatJobsTable(result.jobs));
    });
}
