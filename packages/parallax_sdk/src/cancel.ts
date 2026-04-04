import { getQueue } from "./queue";

/** `null` = job not found; `"terminal"` = already completed/failed; `true` = cancelled */
export type CancelJobOutcome = true | null | "terminal";

export async function cancelJob(id: string): Promise<CancelJobOutcome> {
  const queue = getQueue();
  const job = await queue.getJob(id);

  if (!job) {
    await queue.close();
    return null;
  }

  const state = await job.getState();

  if (state === "completed" || state === "failed") {
    await queue.close();
    return "terminal";
  }

  await queue.removeAsync(id);
  await queue.close();
  return true;
}
