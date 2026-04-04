import { getQueue } from "./queue";
import type { ParallaxJobData } from "./jobs";

export async function submitJob(data: ParallaxJobData): Promise<string> {
  const queue = getQueue();

  const job = await queue.add("pipeline", data, {
    attempts: 1,
    timeout: 30 * 60 * 1000,
  });

  const jobId = String(job.id);

  Bun.spawn(["bun", "packages/parallax_cli/src/_run.ts", jobId], {
    stdin: "ignore",
    stdout: "ignore",
    stderr: "ignore",
    detached: true,
  });

  await queue.close();

  return jobId;
}
