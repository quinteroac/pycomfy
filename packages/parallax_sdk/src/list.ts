import os from "os";
import path from "path";
import { Database } from "bun:sqlite";
import { unpack } from "msgpackr";
import type { ParallaxJobData } from "./jobs";

export type JobStatusValue = "waiting" | "active" | "completed" | "failed";

export interface JobSummary {
  id: string;
  status: JobStatusValue;
  progress: number;
  model: string;
  action: string;
  media: string;
  createdAt: number;
  /** Wall-clock duration in ms (finishedOn − processedOn), or null when not yet available. */
  duration: number | null;
}

export interface JobListResult {
  jobs: JobSummary[];
  counts: {
    waiting: number;
    active: number;
    completed: number;
    failed: number;
  };
}

function mapSqliteState(state: string): JobStatusValue {
  if (state === "active") return "active";
  if (state === "completed") return "completed";
  // waiting / delayed / prioritized / waiting-children all surface as waiting
  return "waiting";
}

const DB_PATH = path.join(os.homedir(), ".config", "parallax", "jobs.db");

export async function listJobs(opts?: {
  status?: JobStatusValue;
  limit?: number;
}): Promise<JobListResult> {
  const limit = opts?.limit ?? 50;

  // Query SQLite directly so state reflects reality regardless of which
  // in-memory QueueManager instance is (or isn't) running.
  const db = new Database(DB_PATH, { readonly: true, create: false });

  try {
    // --- jobs (waiting / active / completed) ---
    const sqliteStates =
      opts?.status === "failed"
        ? []
        : opts?.status
          ? [opts.status]
          : ["waiting", "delayed", "active", "completed"];

    const placeholders = sqliteStates.map(() => "?").join(", ");
    const jobRows: Array<{
      id: string;
      data: Uint8Array | null;
      state: string;
      progress: number | null;
      created_at: number;
      started_at: number | null;
      completed_at: number | null;
    }> =
      sqliteStates.length > 0
        ? (db
            .query(
              `SELECT id, data, state, progress, created_at, started_at, completed_at
               FROM jobs
               WHERE queue = 'parallax' AND state IN (${placeholders})
                 AND id NOT IN (SELECT job_id FROM dlq WHERE queue = 'parallax')
               ORDER BY created_at DESC
               LIMIT ${limit}`,
            )
            .all(...(sqliteStates as string[])) as any[])
        : [];

    // --- DLQ entries (failed jobs) ---
    const includeFailed = !opts?.status || opts.status === "failed";
    const dlqRows: Array<{ job_id: string; entry: Uint8Array | null; entered_at: number }> =
      includeFailed
        ? (db
            .query(
              `SELECT job_id, entry, entered_at FROM dlq
               WHERE queue = 'parallax'
               ORDER BY entered_at DESC
               LIMIT ${limit}`,
            )
            .all() as any[])
        : [];

    // --- counts ---
    const countRow = db
      .query(
        `SELECT
           SUM(CASE WHEN state IN ('waiting','delayed') THEN 1 ELSE 0 END) AS waiting,
           SUM(CASE WHEN state = 'active'    THEN 1 ELSE 0 END) AS active,
           SUM(CASE WHEN state = 'completed' THEN 1 ELSE 0 END) AS completed
         FROM jobs WHERE queue = 'parallax'`,
      )
      .get() as { waiting: number; active: number; completed: number } | null;

    const failedCount = db
      .query(`SELECT COUNT(*) AS n FROM dlq WHERE queue = 'parallax'`)
      .get() as { n: number } | null;

    // --- decode msgpack job data (bunqueue uses msgpackr for the data column) ---
    function decodeData(buf: Uint8Array | null): Partial<ParallaxJobData> {
      if (!buf) return {};
      try {
        return (unpack(buf) as any) ?? {};
      } catch {
        return {};
      }
    }

    const summaries: JobSummary[] = [];

    for (const row of jobRows) {
      const data = decodeData(row.data);
      summaries.push({
        id: row.id,
        status: mapSqliteState(row.state),
        progress: row.progress ?? 0,
        model: data.model ?? "",
        action: data.action ?? "",
        media: data.media ?? "",
        createdAt: row.created_at,
        duration:
          row.started_at != null && row.completed_at != null
            ? row.completed_at - row.started_at
            : null,
      });
    }

    for (const row of dlqRows) {
      // entry is msgpack of { job, error }; we only need job.data fields
      let model = "";
      let action = "";
      let media = "";
      try {
        const entry = (unpack(row.entry!) as any) ?? {};
        const jobData: Partial<ParallaxJobData> = entry?.job?.data ?? {};
        model = jobData.model ?? "";
        action = jobData.action ?? "";
        media = jobData.media ?? "";
      } catch { /* best-effort */ }

      summaries.push({
        id: row.job_id,
        status: "failed",
        progress: 0,
        model,
        action,
        media,
        createdAt: row.entered_at,
        duration: null,
      });
    }

    summaries.sort((a, b) => b.createdAt - a.createdAt);

    return {
      jobs: summaries.slice(0, limit),
      counts: {
        waiting: countRow?.waiting ?? 0,
        active: countRow?.active ?? 0,
        completed: countRow?.completed ?? 0,
        failed: failedCount?.n ?? 0,
      },
    };
  } finally {
    db.close();
  }
}
