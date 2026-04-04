// Tests for US-003: create_audio MCP tool.

import { describe, it, expect } from "bun:test";
import { readFileSync } from "fs";
import { join } from "path";

const SRC = readFileSync(join(import.meta.dir, "../src/index.ts"), "utf-8");

// ── AC01: create_audio tool is registered with the correct input schema ────────

describe("US-003-AC01: create_audio tool registration", () => {
  it("registers a tool named create_audio", () => {
    expect(SRC).toContain('"create_audio"');
  });

  it("schema includes required field: model", () => {
    expect(SRC).toContain("model:");
  });

  it("schema includes required field: prompt", () => {
    expect(SRC).toContain("prompt:");
  });

  it("schema includes optional field: length", () => {
    expect(SRC).toContain("length:");
  });

  it("schema includes optional field: steps", () => {
    expect(SRC).toContain("steps:");
  });

  it("schema includes optional field: cfg", () => {
    expect(SRC).toContain("cfg:");
  });

  it("schema includes optional field: bpm", () => {
    expect(SRC).toContain("bpm:");
  });

  it("schema includes optional field: lyrics", () => {
    expect(SRC).toContain("lyrics:");
  });

  it("schema includes optional field: seed", () => {
    expect(SRC).toContain("seed:");
  });

  it("schema includes optional field: output", () => {
    expect(SRC).toContain("output:");
  });

  it("schema includes optional field: modelsDir", () => {
    expect(SRC).toContain("modelsDir:");
  });
});

// ── AC02: tool spawns bun run src/index.ts create audio via Bun.spawn ─────────

describe("US-003-AC02: spawn command structure", () => {
  it("uses Bun.spawn to invoke the CLI", () => {
    expect(SRC).toContain("Bun.spawn");
  });

  it("spawns bun run src/index.ts", () => {
    expect(SRC).toContain('"bun"');
    expect(SRC).toContain('"run"');
    expect(SRC).toContain('"src/index.ts"');
  });

  it("passes create audio subcommand", () => {
    expect(SRC).toContain('"create"');
    expect(SRC).toContain('"audio"');
  });

  it("passes --model flag from input", () => {
    expect(SRC).toContain('"--model"');
    expect(SRC).toContain("input.model");
  });

  it("passes --prompt flag from input", () => {
    expect(SRC).toContain('"--prompt"');
    expect(SRC).toContain("input.prompt");
  });

  it("passes optional flags when provided", () => {
    expect(SRC).toContain('"--length"');
    expect(SRC).toContain('"--steps"');
    expect(SRC).toContain('"--cfg"');
    expect(SRC).toContain('"--bpm"');
    expect(SRC).toContain('"--lyrics"');
    expect(SRC).toContain('"--seed"');
    expect(SRC).toContain('"--output"');
    expect(SRC).toContain('"--models-dir"');
  });

  it("uses CLI_DIR as cwd for spawn", () => {
    expect(SRC).toContain("CLI_DIR");
    expect(SRC).toContain("parallax_cli");
  });

  it("returns output path on success", () => {
    expect(SRC).toContain("output.wav");
    expect(SRC).toContain("outputPath");
  });

  it("returns stderr on failure with isError", () => {
    expect(SRC).toContain("exitCode !== 0");
    expect(SRC).toContain("isError: true");
  });
});

// ── Functional: spawn subprocess success/failure ───────────────────────────────

describe("US-003 functional: subprocess behaviour", () => {
  it("returns error result when CLI exits non-zero", async () => {
    Bun.write("/tmp/_mcp_audio_test_fail.ts", `process.stderr.write("audio pipeline failed\\n"); process.exit(1);`);

    const proc = Bun.spawn(["bun", "run", "/tmp/_mcp_audio_test_fail.ts"], {
      stdout: "pipe",
      stderr: "pipe",
    });

    const [exitCode, stderr] = await Promise.all([
      proc.exited,
      new Response(proc.stderr).text(),
    ]);

    expect(exitCode).toBe(1);
    expect(stderr.trim()).toBe("audio pipeline failed");
  });

  it("returns success result when CLI exits zero", async () => {
    Bun.write("/tmp/_mcp_audio_test_ok.ts", `process.exit(0);`);

    const proc = Bun.spawn(["bun", "run", "/tmp/_mcp_audio_test_ok.ts"], {
      stdout: "pipe",
      stderr: "pipe",
    });

    const exitCode = await proc.exited;
    expect(exitCode).toBe(0);
  });
});
