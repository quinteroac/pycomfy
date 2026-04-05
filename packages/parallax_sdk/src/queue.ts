import os from "os";
import path from "path";
import { Queue } from "bunqueue/client";

export type { Queue as Bunqueue };

let instance: Queue | null = null;

export function getQueue(): Queue {
  if (!instance) {
    const dbPath = path.join(os.homedir(), ".config", "parallax", "jobs.db");
    instance = new Queue("parallax", {
      embedded: true,
      dataPath: dbPath,
    });
  }
  return instance;
}

export async function closeQueue(): Promise<void> {
  if (instance) {
    await instance.close();
    instance = null;
  }
}
