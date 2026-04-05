import { getQueue, closeQueue } from "./queue";
import type { ParallaxJobData } from "./jobs";

export async function submitJob(data: ParallaxJobData): Promise<string> {
  const queue = getQueue();

  const job = await queue.add("pipeline", data, {
    attempts: 1,
    timeout: 30 * 60 * 1000,
    durable: true,
  });

  const jobId = String(job.id);
  await closeQueue();
  return jobId;
}
