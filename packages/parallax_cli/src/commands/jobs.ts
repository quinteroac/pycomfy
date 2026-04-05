// jobs command: `parallax jobs list`, `parallax jobs watch <id>`, `parallax jobs status <id>`, and `parallax jobs open <id>`

import { Command } from "commander";
import { listJobs } from "@parallax/sdk/list";
import { getJobStatus } from "@parallax/sdk/status";
import { cancelJob } from "@parallax/sdk/cancel";
import type { JobSummary, JobStatusValue } from "@parallax/sdk/list";
import type { ParallaxJobData, ParallaxJobResult } from "@parallax/sdk/jobs";
import type { ParallaxJobStatus } from "@parallax/sdk/status";

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
    duration: formatDuration(job.duration),
  }));
}

export function formatDuration(durationMs: number | null): string {
  if (durationMs === null) return "—";
  const sec = Math.round(durationMs / 1000);
  if (sec < 60) return `${sec}s`;
  const min = Math.floor(sec / 60);
  const remSec = sec % 60;
  return remSec > 0 ? `${min}m ${remSec}s` : `${min}m`;
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

// ── watch command helpers ────────────────────────────────────────────────────

export const SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

export function formatSpinnerLine(frame: number, step: string, progress: number): string {
  const spinner = SPINNER_FRAMES[frame % SPINNER_FRAMES.length];
  return `${spinner} ${step}… ${progress}%`;
}

export function formatDoneLine(outputPath: string): string {
  return `✔ Done: ${outputPath}`;
}

export function formatFailLine(reason: string): string {
  return `✖ Failed: ${reason}`;
}

export function formatNotFoundMessage(id: string): string {
  return `Job ${id} not found`;
}

async function watchJobAction(id: string): Promise<void> {
  const { getQueue } = await import("@parallax/sdk/queue");
  const { spinner, log } = await import("@clack/prompts");

  const queue = getQueue();

  const initialJob = await queue.getJob(id);
  if (!initialJob) {
    await queue.close();
    log.warn(formatNotFoundMessage(id));
    process.exit(1);
    return;
  }

  const s = spinner();
  s.start("Watching job…");

  while (true) {
    const job = await queue.getJob(id);
    if (!job) break;

    const state = await job.getState();

    if (state === "completed") {
      const result = job.returnvalue as ParallaxJobResult | null;
      const outputPath = result?.outputPath ?? "";
      s.stop("");
      log.message(formatDoneLine(outputPath));
      await queue.close();
      return;
    }

    if (state === "failed") {
      s.stop("");
      log.message(formatFailLine((job as any).failedReason ?? "unknown error"));
      await queue.close();
      process.exit(1);
      return;
    }

    const progress = typeof job.progress === "number" ? job.progress : 0;
    const data = job.data as Partial<ParallaxJobData>;
    const step = data?.action ?? "processing";

    s.message(`${step}… ${progress}%`);

    await new Promise<void>((resolve) => setTimeout(resolve, 500));
  }

  await queue.close();
}

// ── status command helpers ───────────────────────────────────────────────────

function formatTimestamp(ts: number | null): string {
  if (ts === null) return "—";
  return new Date(ts).toISOString();
}

export function formatJobStatus(job: ParallaxJobStatus): string {
  const lines: string[] = [
    `id:         ${job.id}`,
    `status:     ${job.status}`,
    `progress:   ${job.progress}%`,
    `model:      ${job.model ?? "—"}`,
    `action:     ${job.action ?? "—"}`,
  ];
  if (job.error != null) {
    lines.push(`error:      ${job.error}`);
  } else {
    lines.push(`output:     ${job.output ?? "—"}`);
  }
  lines.push(`startedAt:  ${formatTimestamp(job.startedAt)}`);
  lines.push(`finishedAt: ${formatTimestamp(job.finishedAt)}`);
  return lines.join("\n");
}

export function formatJobStatusJson(job: ParallaxJobStatus): string {
  return JSON.stringify({
    id: job.id,
    status: job.status,
    progress: job.progress,
    model: job.model,
    action: job.action,
    output: job.error == null ? job.output : undefined,
    error: job.error ?? undefined,
    startedAt: job.startedAt,
    finishedAt: job.finishedAt,
  });
}

async function statusJobAction(id: string, opts: { json?: boolean }): Promise<void> {
  const job = await getJobStatus(id);
  if (!job) {
    process.stderr.write(`Job ${id} not found\n`);
    process.exit(1);
    return;
  }

  if (opts.json) {
    console.log(formatJobStatusJson(job));
  } else {
    console.log(formatJobStatus(job));
  }

  if (job.status === "failed") {
    process.exit(1);
  }
}

export function formatCancelledMessage(id: string): string {
  return `✔ Job ${id} cancelled`;
}

export function formatAlreadyTerminalMessage(id: string, status: string): string {
  return `Job ${id} is already ${status} — nothing to cancel`;
}

// ── open command helpers ─────────────────────────────────────────────────────

export function formatNotCompletedMessage(id: string, status: string): string {
  return `Job ${id} is not completed yet (status: ${status})`;
}

async function openJobAction(id: string): Promise<void> {
  const job = await getJobStatus(id);

  if (!job) {
    process.stdout.write(formatNotFoundMessage(id) + "\n");
    process.exit(1);
    return;
  }

  if (job.status !== "completed") {
    process.stdout.write(formatNotCompletedMessage(id, job.status) + "\n");
    process.exit(1);
    return;
  }

  const outputPath = job.output ?? "";
  const opener = process.platform === "darwin" ? "open" : "xdg-open";
  Bun.spawn([opener, outputPath]);
}

async function cancelJobAction(id: string): Promise<void> {
  const outcome = await cancelJob(id);

  if (outcome === null) {
    process.stdout.write(formatNotFoundMessage(id) + "\n");
    process.exit(1);
    return;
  }

  if (outcome === "completed" || outcome === "failed") {
    console.log(formatAlreadyTerminalMessage(id, outcome));
    return;
  }

  console.log(formatCancelledMessage(id));
}

export function registerJobs(program: Command): void {
  const jobs = program
    .command("jobs")
    .description("Manage background jobs");

  jobs
    .command("list")
    .description("Show recent jobs (newest first, max 20)")
    .action(async () => {
      const { log } = await import("@clack/prompts");
      const result = await listJobs({ limit: 20 });
      log.message(formatJobsTable(result.jobs));
    });

  jobs
    .command("watch <id>")
    .description("Watch a job until it finishes")
    .action(async (id: string) => {
      await watchJobAction(id);
    });

  jobs
    .command("open <id>")
    .description("Open the output file of a completed job in the default application")
    .action(async (id: string) => {
      await openJobAction(id);
    });

  jobs
    .command("cancel <id>")
    .description("Cancel a running or waiting job")
    .action(async (id: string) => {
      await cancelJobAction(id);
    });

  jobs
    .command("status <id>")
    .description("Print a one-shot status block for a job")
    .option("--json", "Output as JSON")
    .action(async (id: string, opts: { json?: boolean }) => {
      await statusJobAction(id, opts);
    });
}
