import { getQueue } from "./queue";
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

function mapState(state: string): JobStatusValue {
  if (state === "active") return "active";
  if (state === "completed") return "completed";
  if (state === "failed") return "failed";
  return "waiting";
}

const ALL_STATES = ["waiting", "active", "completed", "failed"] as const;

export async function listJobs(opts?: {
  status?: JobStatusValue;
  limit?: number;
}): Promise<JobListResult> {
  const limit = opts?.limit ?? 50;
  const queue = getQueue();

  const [rawJobs, rawCounts] = await Promise.all([
    queue.getJobsAsync({
      state: opts?.status ? [opts.status] : ([...ALL_STATES] as string[]),
    }),
    queue.getJobCountsAsync(),
  ]);

  await queue.close();

  const stateMap = await Promise.all(
    rawJobs.map(async (job) => {
      const state = await job.getState();
      return { job, state };
    }),
  );

  const summaries: JobSummary[] = stateMap
    .sort((a, b) => b.job.timestamp - a.job.timestamp)
    .slice(0, limit)
    .map(({ job, state }) => {
      const data = (job.data ?? {}) as Partial<ParallaxJobData>;
      return {
        id: job.id,
        status: mapState(state),
        progress: typeof job.progress === "number" ? job.progress : 0,
        model: data.model ?? "",
        action: data.action ?? "",
        media: data.media ?? "",
        createdAt: job.timestamp,
      };
    });

  return {
    jobs: summaries,
    counts: {
      waiting: rawCounts.waiting ?? 0,
      active: rawCounts.active ?? 0,
      completed: rawCounts.completed ?? 0,
      failed: rawCounts.failed ?? 0,
    },
  };
}
