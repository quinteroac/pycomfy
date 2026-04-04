import { getQueue } from "./queue";

export type ParallaxJobStatus = {
  id: string;
  status: "waiting" | "active" | "completed" | "failed";
  progress: number;
  output: string | null;
  error: string | null;
  createdAt: number;
  startedAt: number | null;
  finishedAt: number | null;
};

function mapState(state: string): "waiting" | "active" | "completed" | "failed" {
  if (state === "active") return "active";
  if (state === "completed") return "completed";
  if (state === "failed") return "failed";
  return "waiting";
}

export async function getJobStatus(id: string): Promise<ParallaxJobStatus | null> {
  const queue = getQueue();
  const job = await queue.getJob(id);
  await queue.close();
  if (!job) return null;

  const state = await job.getState();
  return {
    id: job.id,
    status: mapState(state),
    progress: typeof job.progress === "number" ? job.progress : 0,
    output: job.returnvalue != null ? String(job.returnvalue) : null,
    error: job.failedReason ?? null,
    createdAt: job.timestamp,
    startedAt: job.processedOn ?? null,
    finishedAt: job.finishedOn ?? null,
  };
}
