import { getQueue, closeQueue } from "./queue";
import type { ParallaxJobData } from "./jobs";

export type ParallaxJobStatus = {
  id: string;
  status: "waiting" | "active" | "completed" | "failed";
  progress: number;
  model: string | null;
  action: string | null;
  media: string | null;
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
  await closeQueue();
  if (!job) return null;

  const state = await job.getState();
  const data = (job.data ?? {}) as Partial<ParallaxJobData>;
  const result = job.returnvalue as { outputPath?: string } | null;
  return {
    id: job.id,
    status: mapState(state),
    progress: typeof job.progress === "number" ? job.progress : 0,
    model: data.model ?? null,
    action: data.action ?? null,
    media: data.media ?? null,
    output: result?.outputPath ?? null,
    error: (job as any).failedReason ?? null,
    createdAt: job.timestamp,
    startedAt: job.processedOn ?? null,
    finishedAt: job.finishedOn ?? null,
  };
}
