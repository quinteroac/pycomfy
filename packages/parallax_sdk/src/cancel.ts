import { getQueue, closeQueue } from "./queue";

/** `null` = job not found; `"completed"` or `"failed"` = already in terminal state; `true` = cancelled */
export type CancelJobOutcome = true | null | "completed" | "failed";

export async function cancelJob(id: string): Promise<CancelJobOutcome> {
  const queue = getQueue();
  const job = await queue.getJob(id);

  if (!job) {
    await closeQueue();
    return null;
  }

  const state = await job.getState();

  if (state === "completed" || state === "failed") {
    await closeQueue();
    return state;
  }

  await queue.removeAsync(id);
  await closeQueue();
  return true;
}
