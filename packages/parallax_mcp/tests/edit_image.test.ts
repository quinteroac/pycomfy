// Tests for US-004: edit_image MCP tool.

import { describe, it, expect } from "bun:test";
import { readFileSync } from "fs";
import { join } from "path";

const SRC = readFileSync(join(import.meta.dir, "../src/index.ts"), "utf-8");

// ── AC01: edit_image tool is registered with the correct input schema ─────────

describe("US-004-AC01: edit_image tool registration", () => {
  it("registers a tool named edit_image", () => {
    expect(SRC).toContain('"edit_image"');
  });

  it("schema includes required field: model", () => {
    expect(SRC).toContain("model:");
  });

  it("schema includes required field: prompt", () => {
    expect(SRC).toContain("prompt:");
  });

  it("schema includes required field: input", () => {
    expect(SRC).toContain("input:");
  });

  it("schema includes optional field: subjectImage", () => {
    expect(SRC).toContain("subjectImage:");
  });

  it("schema includes optional field: width", () => {
    expect(SRC).toContain("width:");
  });

  it("schema includes optional field: height", () => {
    expect(SRC).toContain("height:");
  });

  it("schema includes optional field: steps", () => {
    expect(SRC).toContain("steps:");
  });

  it("schema includes optional field: cfg", () => {
    expect(SRC).toContain("cfg:");
  });

  it("schema includes optional field: seed", () => {
    expect(SRC).toContain("seed:");
  });

  it("schema includes optional field: output", () => {
    expect(SRC).toContain("output:");
  });

  it("schema includes optional field: image2", () => {
    expect(SRC).toContain("image2:");
  });

  it("schema includes optional field: image3", () => {
    expect(SRC).toContain("image3:");
  });

  it("schema includes optional field: noLora", () => {
    expect(SRC).toContain("noLora:");
  });

  it("schema includes optional field: modelsDir", () => {
    expect(SRC).toContain("modelsDir:");
  });
});

// ── AC02: tool spawns bun run src/index.ts edit image via Bun.spawn ────────────

describe("US-004-AC02: spawn command structure", () => {
  it("uses Bun.spawn to invoke the CLI", () => {
    expect(SRC).toContain("Bun.spawn");
  });

  it("spawns bun run src/index.ts", () => {
    expect(SRC).toContain('"bun"');
    expect(SRC).toContain('"run"');
    expect(SRC).toContain('"src/index.ts"');
  });

  it("passes edit image subcommand", () => {
    expect(SRC).toContain('"edit"');
    expect(SRC).toContain('"image"');
  });

  it("passes --model flag from input", () => {
    expect(SRC).toContain('"--model"');
    expect(SRC).toContain("input.model");
  });

  it("passes --prompt flag from input", () => {
    expect(SRC).toContain('"--prompt"');
    expect(SRC).toContain("input.prompt");
  });

  it("passes --input flag from input", () => {
    expect(SRC).toContain('"--input"');
    expect(SRC).toContain("input.input");
  });

  it("passes optional flags when provided", () => {
    expect(SRC).toContain('"--subject-image"');
    expect(SRC).toContain('"--width"');
    expect(SRC).toContain('"--height"');
    expect(SRC).toContain('"--steps"');
    expect(SRC).toContain('"--cfg"');
    expect(SRC).toContain('"--seed"');
    expect(SRC).toContain('"--output"');
    expect(SRC).toContain('"--image2"');
    expect(SRC).toContain('"--image3"');
    expect(SRC).toContain('"--no-lora"');
    expect(SRC).toContain('"--models-dir"');
  });

  it("uses CLI_DIR as cwd for spawn", () => {
    expect(SRC).toContain("CLI_DIR");
    expect(SRC).toContain("parallax_cli");
  });

  it("resolves output path before spawn", () => {
    expect(SRC).toContain("outputPath");
    expect(SRC).toContain("resolve(");
    expect(SRC).toContain("output.png");
  });

  it("returns output path on success", () => {
    expect(SRC).toContain('content: [{ type: "text", text: outputPath }]');
  });

  it("returns error with stderr on non-zero exit", () => {
    expect(SRC).toContain("exitCode !== 0");
    expect(SRC).toContain("stderr");
    expect(SRC).toContain("isError: true");
  });
});

// ── Functional: subprocess spawn behaviour ────────────────────────────────────

describe("US-004 functional: subprocess behaviour", () => {
  it("returns error result when CLI exits non-zero", async () => {
    Bun.write("/tmp/_mcp_edit_test_fail.ts", `process.stderr.write("edit failed\\n"); process.exit(1);`);

    const proc = Bun.spawn(["bun", "run", "/tmp/_mcp_edit_test_fail.ts"], {
      stdout: "pipe",
      stderr: "pipe",
    });

    const [exitCode, stderr] = await Promise.all([
      proc.exited,
      new Response(proc.stderr).text(),
    ]);

    expect(exitCode).toBe(1);
    expect(stderr.trim()).toBe("edit failed");
  });

  it("returns success result when CLI exits zero", async () => {
    Bun.write("/tmp/_mcp_edit_test_ok.ts", `process.exit(0);`);

    const proc = Bun.spawn(["bun", "run", "/tmp/_mcp_edit_test_ok.ts"], {
      stdout: "pipe",
      stderr: "pipe",
    });

    const exitCode = await proc.exited;
    expect(exitCode).toBe(0);
  });
});
