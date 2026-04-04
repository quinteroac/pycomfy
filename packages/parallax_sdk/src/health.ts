import { getQueue } from "./queue";

export interface QueueStats {
  waiting: number;
  active: number;
  completed: number;
  failed: number;
}

export async function getQueueStats(): Promise<QueueStats> {
  const queue = getQueue();
  const counts = await queue.getJobCountsAsync();
  await queue.close();

  return {
    waiting: counts.waiting ?? 0,
    active: counts.active ?? 0,
    completed: counts.completed ?? 0,
    failed: counts.failed ?? 0,
  };
}
