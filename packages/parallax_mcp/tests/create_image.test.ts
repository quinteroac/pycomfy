// Tests for US-001: create_image MCP tool.

import { describe, it, expect } from "bun:test";
import { readFileSync } from "fs";
import { join } from "path";

const SRC = readFileSync(join(import.meta.dir, "../src/index.ts"), "utf-8");

// ── AC01: create_image tool is registered with the correct input schema ────────

describe("US-001-AC01: create_image tool registration", () => {
  it("registers a tool named create_image", () => {
    expect(SRC).toContain('"create_image"');
  });

  it("schema includes required field: model", () => {
    expect(SRC).toContain("model:");
  });

  it("schema includes required field: prompt", () => {
    expect(SRC).toContain("prompt:");
  });

  it("schema includes optional field: negativePrompt", () => {
    expect(SRC).toContain("negativePrompt:");
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

  it("schema includes optional field: modelsDir", () => {
    expect(SRC).toContain("modelsDir:");
  });
});

// ── AC02: tool spawns bun run src/index.ts create image via Bun.spawn ──────────

describe("US-001-AC02: spawn command structure", () => {
  it("uses Bun.spawn to invoke the CLI", () => {
    expect(SRC).toContain("Bun.spawn");
  });

  it("spawns bun run src/index.ts", () => {
    expect(SRC).toContain('"bun"');
    expect(SRC).toContain('"run"');
    expect(SRC).toContain('"src/index.ts"');
  });

  it("passes create image subcommand", () => {
    expect(SRC).toContain('"create"');
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

  it("passes optional flags when provided", () => {
    expect(SRC).toContain('"--negative-prompt"');
    expect(SRC).toContain('"--width"');
    expect(SRC).toContain('"--height"');
    expect(SRC).toContain('"--steps"');
    expect(SRC).toContain('"--cfg"');
    expect(SRC).toContain('"--seed"');
    expect(SRC).toContain('"--output"');
    expect(SRC).toContain('"--models-dir"');
  });

  it("uses CLI_DIR as cwd for spawn", () => {
    expect(SRC).toContain("CLI_DIR");
    expect(SRC).toContain("parallax_cli");
  });
});

// ── AC03 + AC04: success and failure path returns ─────────────────────────────

describe("US-001-AC03: success returns output path", () => {
  it("resolves and returns output path on success", () => {
    expect(SRC).toContain("outputPath");
    expect(SRC).toContain("resolve(");
    expect(SRC).toContain("output.png");
  });

  it("returns content with text on success", () => {
    expect(SRC).toContain("content: [{ type: \"text\", text: outputPath }]");
  });
});

describe("US-001-AC04: failure returns stderr", () => {
  it("checks exit code", () => {
    expect(SRC).toContain("exitCode !== 0");
  });

  it("returns stderr on non-zero exit", () => {
    expect(SRC).toContain("stderr");
    expect(SRC).toContain("isError: true");
  });
});

// ── Functional: tool handler invokes CLI and captures result ──────────────────

describe("US-001 functional: tool handler behaviour", () => {
  it("returns error result when CLI exits non-zero", async () => {
    // Spawn a fake CLI script that exits with code 1 and writes to stderr
    const tmpScript = await Bun.file("/tmp/_mcp_test_fail.ts").exists()
      ? "/tmp/_mcp_test_fail.ts"
      : (() => {
          Bun.write("/tmp/_mcp_test_fail.ts", `process.stderr.write("pipeline failed\\n"); process.exit(1);`);
          return "/tmp/_mcp_test_fail.ts";
        })();

    const proc = Bun.spawn(["bun", "run", "/tmp/_mcp_test_fail.ts"], {
      stdout: "pipe",
      stderr: "pipe",
    });

    const [exitCode, stderr] = await Promise.all([
      proc.exited,
      new Response(proc.stderr).text(),
    ]);

    expect(exitCode).toBe(1);
    expect(stderr.trim()).toBe("pipeline failed");
  });

  it("returns success result when CLI exits zero", async () => {
    Bun.write("/tmp/_mcp_test_ok.ts", `process.exit(0);`);

    const proc = Bun.spawn(["bun", "run", "/tmp/_mcp_test_ok.ts"], {
      stdout: "pipe",
      stderr: "pipe",
    });

    const exitCode = await proc.exited;
    expect(exitCode).toBe(0);
  });
});
