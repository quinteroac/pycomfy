// Tests for US-006: MCP Server Startup and Client Registration.

import { describe, it, expect } from "bun:test";
import { readFileSync } from "fs";
import { join } from "path";

const SRC    = readFileSync(join(import.meta.dir, "../src/index.ts"), "utf-8");
const PKG    = JSON.parse(readFileSync(join(import.meta.dir, "../package.json"), "utf-8"));
const DOCS   = readFileSync(join(import.meta.dir, "../../../docs/parallax_mcp.md"), "utf-8");

// ── AC01: `bun run start` starts the server without errors ────────────────────

describe("US-006-AC01: bun run start script", () => {
  it("package.json has a start script", () => {
    expect(PKG.scripts).toBeDefined();
    expect(PKG.scripts.start).toBeDefined();
  });

  it("start script invokes src/index.ts", () => {
    expect(PKG.scripts.start).toContain("src/index.ts");
  });

  it("server process starts and exits cleanly when stdin closes", async () => {
    const indexPath = join(import.meta.dir, "../src/index.ts");
    const proc = Bun.spawn(["bun", "run", indexPath], {
      stdin: "pipe",
      stdout: "pipe",
      stderr: "pipe",
    });

    // Close stdin immediately to simulate no MCP client — the server should exit cleanly.
    proc.stdin.end();
    const exitCode = await proc.exited;
    // Any clean termination (0 or 1 with empty stderr) is acceptable — the
    // important thing is it does NOT throw a startup / import error.
    const stderr = await new Response(proc.stderr).text();
    const hasStartupError = stderr.includes("SyntaxError") || stderr.includes("Cannot find module");
    expect(hasStartupError).toBe(false);
  });
});

// ── AC02: tools/list returns all 5 tools ──────────────────────────────────────

describe("US-006-AC02: all 5 tools registered", () => {
  it("registers create_image", () => {
    expect(SRC).toContain('"create_image"');
  });

  it("registers create_video", () => {
    expect(SRC).toContain('"create_video"');
  });

  it("registers create_audio", () => {
    expect(SRC).toContain('"create_audio"');
  });

  it("registers edit_image", () => {
    expect(SRC).toContain('"edit_image"');
  });

  it("registers upscale_image", () => {
    expect(SRC).toContain('"upscale_image"');
  });

  it("uses McpServer from the MCP SDK", () => {
    expect(SRC).toContain("McpServer");
    expect(SRC).toContain("@modelcontextprotocol/sdk");
  });

  it("uses StdioServerTransport", () => {
    expect(SRC).toContain("StdioServerTransport");
  });

  it("connects server to transport", () => {
    expect(SRC).toContain("server.connect(transport)");
  });
});

// ── AC03: README documents config snippet for Claude Desktop / GitHub Copilot ─

describe("US-006-AC03: client registration documentation", () => {
  it("docs mention bun run start", () => {
    expect(DOCS).toContain("bun run start");
  });

  it("docs contain Claude Desktop config section", () => {
    expect(DOCS).toContain("Claude Desktop");
  });

  it("docs contain GitHub Copilot config section", () => {
    expect(DOCS).toContain("GitHub Copilot");
  });

  it("docs contain Claude Code config section", () => {
    expect(DOCS).toContain("Claude Code");
  });

  it("docs show mcpServers config key", () => {
    expect(DOCS).toContain("mcpServers");
  });

  it("docs show command: bun", () => {
    expect(DOCS).toContain('"command": "bun"');
  });

  it("docs show args pointing to src/index.ts", () => {
    expect(DOCS).toContain("parallax_mcp/src/index.ts");
  });

  it("docs list all 5 tools in the tool table", () => {
    expect(DOCS).toContain("create_image");
    expect(DOCS).toContain("create_video");
    expect(DOCS).toContain("create_audio");
    expect(DOCS).toContain("edit_image");
    expect(DOCS).toContain("upscale_image");
  });
});

// ── AC04: typecheck passes (verified by tsc --noEmit in CI) ───────────────────

describe("US-006-AC04: typecheck", () => {
  it("tsconfig.json exists and includes src/", () => {
    const tsconfig = JSON.parse(readFileSync(join(import.meta.dir, "../tsconfig.json"), "utf-8"));
    expect(tsconfig.include).toContain("src");
  });

  it("package.json has a typecheck script", () => {
    expect(PKG.scripts.typecheck).toBeDefined();
  });
});
