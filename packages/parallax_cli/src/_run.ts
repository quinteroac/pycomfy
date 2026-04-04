/**
 * Detached worker process: picks up a specific job by ID, runs the Python
 * pipeline, streams NDJSON progress, and marks the job completed or failed.
 *
 * Invoked as: bun packages/parallax_cli/src/_run.ts <jobId>
 */

import os from "os";
import path from "path";
import { join } from "path";
import { Bunqueue } from "bunqueue/client";
import type { Job } from "bunqueue/client";
import type { ParallaxJobData, ParallaxJobResult, PythonProgress } from "@parallax/sdk";

const jobId = process.argv[2];
if (!jobId) {
  console.error("Usage: _run.ts <jobId>");
  process.exit(1);
}

const dbPath = path.join(os.homedir(), ".config", "parallax", "jobs.db");

const queue = new Bunqueue<ParallaxJobData, ParallaxJobResult>("parallax", {
  routes: {
    pipeline: async (job: Job<ParallaxJobData>) => {
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
          if (typeof event.pct === "number") {
            await job.updateProgress(event.pct);
          }
          if (event.output) {
            outputPath = event.output;
          }
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
    },
  },
  embedded: true,
  dataPath: dbPath,
});

// Close the queue once the job finishes (completed or failed)
queue.once("completed", async () => {
  await queue.close();
});

queue.once("failed", async () => {
  await queue.close();
});
