import { describe, it, expect } from "bun:test";
import { join } from "path";
import { SUPPORTED_CLIENTS, CLIENT_LABELS } from "../../src/commands/mcp";

const CLI = join(import.meta.dir, "../../src/index.ts");

async function runCLI(
  args: string[],
): Promise<{ stdout: string; stderr: string; exitCode: number }> {
  const proc = Bun.spawn(["bun", "run", CLI, ...args], {
    stdout: "pipe",
    stderr: "pipe",
  });
  const [stdout, stderr, exitCode] = await Promise.all([
    new Response(proc.stdout).text(),
    new Response(proc.stderr).text(),
    proc.exited,
  ]);
  return { stdout, stderr, exitCode };
}

// ── AC01/AC02: command registration ──────────────────────────────────────────

describe("parallax mcp install — command registration (US-001-AC01/AC02)", () => {
  it("AC01: `parallax mcp --help` lists the install subcommand", async () => {
    const { stdout, exitCode } = await runCLI(["mcp", "--help"]);
    expect(exitCode).toBe(0);
    expect(stdout).toContain("install");
  });

  it("AC01: `parallax mcp install --help` exits 0 and shows description", async () => {
    const { stdout, exitCode } = await runCLI(["mcp", "install", "--help"]);
    expect(exitCode).toBe(0);
    expect(stdout.toLowerCase()).toContain("mcp");
  });

  it("AC01: `parallax mcp install --help` lists --non-interactive flag", async () => {
    const { stdout, exitCode } = await runCLI(["mcp", "install", "--help"]);
    expect(exitCode).toBe(0);
    expect(stdout).toContain("--non-interactive");
  });

  it("AC01: `parallax mcp install --help` lists --clients flag", async () => {
    const { stdout, exitCode } = await runCLI(["mcp", "install", "--help"]);
    expect(exitCode).toBe(0);
    expect(stdout).toContain("--clients");
  });
});

// ── AC02: supported clients list ─────────────────────────────────────────────

describe("parallax mcp install — supported clients (US-001-AC02)", () => {
  it("AC02: SUPPORTED_CLIENTS contains claude", () => {
    expect(SUPPORTED_CLIENTS).toContain("claude");
  });

  it("AC02: SUPPORTED_CLIENTS contains gemini", () => {
    expect(SUPPORTED_CLIENTS).toContain("gemini");
  });

  it("AC02: SUPPORTED_CLIENTS contains github-copilot", () => {
    expect(SUPPORTED_CLIENTS).toContain("github-copilot");
  });

  it("AC02: SUPPORTED_CLIENTS contains codex", () => {
    expect(SUPPORTED_CLIENTS).toContain("codex");
  });

  it("AC02: SUPPORTED_CLIENTS has exactly 4 entries", () => {
    expect(SUPPORTED_CLIENTS.length).toBe(4);
  });
});

// ── AC03: multi-select — non-interactive mode (no TTY in test subprocess) ────

describe("parallax mcp install --non-interactive (US-001-AC03)", () => {
  it("AC03: installs all clients when --clients is omitted", async () => {
    const { stdout, exitCode } = await runCLI(["mcp", "install", "--non-interactive"]);
    expect(exitCode).toBe(0);
    expect(stdout).toContain("Claude");
    expect(stdout).toContain("Gemini");
    expect(stdout).toContain("GitHub Copilot");
    expect(stdout).toContain("Codex");
  });

  it("AC03: installs only claude when --clients claude", async () => {
    const { stdout, exitCode } = await runCLI([
      "mcp",
      "install",
      "--non-interactive",
      "--clients",
      "claude",
    ]);
    expect(exitCode).toBe(0);
    expect(stdout).toContain("Claude");
  });

  it("AC03: installs claude and gemini when --clients claude,gemini", async () => {
    const { stdout, exitCode } = await runCLI([
      "mcp",
      "install",
      "--non-interactive",
      "--clients",
      "claude,gemini",
    ]);
    expect(exitCode).toBe(0);
    expect(stdout).toContain("Claude");
    expect(stdout).toContain("Gemini");
  });

  it("AC03: installs all four clients when all listed in --clients", async () => {
    const { stdout, exitCode } = await runCLI([
      "mcp",
      "install",
      "--non-interactive",
      "--clients",
      "claude,gemini,github-copilot,codex",
    ]);
    expect(exitCode).toBe(0);
    expect(stdout).toContain("Claude");
    expect(stdout).toContain("Gemini");
    expect(stdout).toContain("GitHub Copilot");
    expect(stdout).toContain("Codex");
  });

  it("AC03: no-TTY subprocess without --non-interactive flag falls back to non-interactive", async () => {
    // Bun.spawn pipes stdout → subprocess sees no TTY → auto non-interactive
    const { stdout, exitCode } = await runCLI(["mcp", "install"]);
    expect(exitCode).toBe(0);
    expect(stdout).toContain("[parallax] MCP clients configured");
  });

  it("AC03: output confirms clients configured header", async () => {
    const { stdout, exitCode } = await runCLI(["mcp", "install", "--non-interactive"]);
    expect(exitCode).toBe(0);
    expect(stdout).toContain("[parallax] MCP clients configured");
  });
});

// ── AC04: type exports are correct ───────────────────────────────────────────

describe("parallax mcp — module exports (US-001-AC04)", () => {
  it("AC04: CLIENT_LABELS keys match SUPPORTED_CLIENTS", () => {
    for (const client of SUPPORTED_CLIENTS) {
      expect(CLIENT_LABELS).toHaveProperty(client);
      expect(typeof CLIENT_LABELS[client]).toBe("string");
    }
  });

  it("AC04: CLIENT_LABELS['github-copilot'] is 'GitHub Copilot'", () => {
    expect(CLIENT_LABELS["github-copilot"]).toBe("GitHub Copilot");
  });
});
