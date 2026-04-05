/**
 * Serial inference daemon — spawned once by @parallax/ms on startup.
 *
 * Processes "pipeline" jobs one at a time. A promise-chain mutex serialises
 * execution so a second job never starts until the first Python subprocess
 * has exited, regardless of bunqueue's internal dispatch behaviour.
 *
 * Invoked as: bun packages/parallax_cli/src/_worker.ts
 */

import os from "os";
import path from "path";
import { join } from "path";
import { Bunqueue } from "bunqueue/client";
import type { Job } from "bunqueue/client";
import type { ParallaxJobData, ParallaxJobResult, PythonProgress } from "@parallax/sdk";

const dbPath = path.join(os.homedir(), ".config", "parallax", "jobs.db");

// Mutex: serialises job execution even if bunqueue dispatches concurrently.
let lock: Promise<void> = Promise.resolve();

async function runPipeline(job: Job<ParallaxJobData>): Promise<ParallaxJobResult> {
  const { uvPath = "uv", scriptBase, script, args } = job.data;

  const proc = Bun.spawn(
    [uvPath, "run", "python", join(scriptBase, script), ...args],
    { stdout: "pipe", stderr: "pipe" },
  );

  // Collect stderr for error reporting
  let stderrOutput = "";
  const stderrReader = proc.stderr.getReader();
  const stderrDecoder = new TextDecoder();
  (async () => {
    while (true) {
      const { done, value } = await stderrReader.read();
      if (done) break;
      stderrOutput += stderrDecoder.decode(value);
    }
  })();

  // Read stdout line by line; parse PythonProgress events
  const stdoutReader = proc.stdout.getReader();
  const stdoutDecoder = new TextDecoder();
  let stdoutBuffer = "";
  let outputPath = "";

  while (true) {
    const { done, value } = await stdoutReader.read();
    if (done) break;
    stdoutBuffer += stdoutDecoder.decode(value);

    const lines = stdoutBuffer.split("\n");
    stdoutBuffer = lines.pop() ?? "";

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      try {
        const event = JSON.parse(trimmed) as PythonProgress;
        if (typeof event.pct === "number") {
          await job.updateProgress(event.pct);
        }
        if (event.output) {
          outputPath = event.output;
        }
      } catch {
        // Not a JSON progress line — ignore
      }
    }
  }

  // Process any remaining buffered content
  if (stdoutBuffer.trim()) {
    try {
      const event = JSON.parse(stdoutBuffer.trim()) as PythonProgress;
      if (typeof event.pct === "number") await job.updateProgress(event.pct);
      if (event.output) outputPath = event.output;
    } catch {
      // Ignore non-JSON trailing output
    }
  }

  const exitCode = await proc.exited;

  if (exitCode !== 0) {
    const stderrTrimmed = stderrOutput.trim();
    const detail = stderrTrimmed ? `\n${stderrTrimmed}` : "";
    throw new Error(`Pipeline failed (exit ${exitCode})${detail}`);
  }

  return { outputPath };
}

const queue = new Bunqueue<ParallaxJobData, ParallaxJobResult>("parallax", {
  routes: {
    pipeline: (job: Job<ParallaxJobData>) => {
      // Chain onto the lock so jobs execute strictly one after another.
      const run = lock.then(() => runPipeline(job));
      lock = run.then(() => {}, () => {});
      return run;
    },
  },
  embedded: true,
  dataPath: dbPath,
});

process.on("SIGTERM", async () => {
  await queue.close();
  process.exit(0);
});

process.on("SIGINT", async () => {
  await queue.close();
  process.exit(0);
});
