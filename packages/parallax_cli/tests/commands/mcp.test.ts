import { describe, it, expect } from "bun:test";
import { join, dirname } from "path";
import { mkdirSync, rmSync, existsSync, readFileSync, writeFileSync } from "fs";
import { tmpdir } from "os";
import { SUPPORTED_CLIENTS, CLIENT_LABELS } from "../../src/commands/mcp";
import { getConfigPath, applyClientConfig, getMcpServerEntry } from "../../src/mcp_config";

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

// ── US-002 AC01: config path resolution per OS ────────────────────────────────

describe("getConfigPath — OS-specific paths (US-002-AC01)", () => {
  const home = "/test/home";
  const appData = "C:\\Users\\test\\AppData\\Roaming";

  it("AC01: claude → Linux path ends with .config/Claude/claude_desktop_config.json", () => {
    const p = getConfigPath("claude", "linux", home);
    expect(p).toContain("Claude");
    expect(p).toContain("claude_desktop_config.json");
    expect(p).toContain(".config");
  });

  it("AC01: claude → macOS path includes Library/Application Support", () => {
    const p = getConfigPath("claude", "darwin", home);
    expect(p).toContain("Library");
    expect(p).toContain("Application Support");
    expect(p).toContain("claude_desktop_config.json");
  });

  it("AC01: claude → Windows path uses APPDATA + Claude", () => {
    const p = getConfigPath("claude", "win32", home, appData);
    expect(p).toContain("Claude");
    expect(p).toContain("claude_desktop_config.json");
    expect(p.startsWith(appData)).toBe(true);
  });

  it("AC01: gemini → ~/.gemini/settings.json on all platforms", () => {
    for (const platform of ["linux", "darwin", "win32"]) {
      const p = getConfigPath("gemini", platform, home);
      expect(p).toContain(".gemini");
      expect(p).toContain("settings.json");
    }
  });

  it("AC01: github-copilot → ~/.copilot/mcp-config.json on all platforms", () => {
    for (const platform of ["linux", "darwin", "win32"]) {
      const p = getConfigPath("github-copilot", platform, home);
      expect(p).toContain(".copilot");
      expect(p).toContain("mcp-config.json");
    }
  });

  it("AC01: codex → ~/.cursor/mcp.json on Linux", () => {
    const p = getConfigPath("codex", "linux", home);
    expect(p).toContain(".cursor");
    expect(p).toContain("mcp.json");
  });

  it("AC01: codex → ~/.cursor/mcp.json on macOS", () => {
    const p = getConfigPath("codex", "darwin", home);
    expect(p).toContain(".cursor");
    expect(p).toContain("mcp.json");
  });
});

// ── US-002 AC02–AC05: applyClientConfig writes correct JSON ───────────────────

function makeTempDir(): string {
  const dir = join(tmpdir(), `parallax-test-${Date.now()}-${Math.random().toString(36).slice(2)}`);
  mkdirSync(dir, { recursive: true });
  return dir;
}

const SERVER_ENTRY = { command: "bun", args: ["run", "/fake/path/index.ts"] };

describe("applyClientConfig — config file writing (US-002-AC02 to AC05)", () => {
  it("AC02: claude — creates claude_desktop_config.json with mcpServers.parallax", () => {
    const tmp = makeTempDir();
    const configPath = join(tmp, "claude_desktop_config.json");
    const result = applyClientConfig("claude", SERVER_ENTRY, configPath);

    expect(result.success).toBe(true);
    expect(existsSync(configPath)).toBe(true);
    const json = JSON.parse(readFileSync(configPath, "utf8"));
    expect(json.mcpServers?.parallax?.command).toBe("bun");
    rmSync(tmp, { recursive: true });
  });

  it("AC02: claude — preserves existing keys when config already exists", () => {
    const tmp = makeTempDir();
    const configPath = join(tmp, "claude_desktop_config.json");
    writeFileSync(configPath, JSON.stringify({ theme: "dark" }), "utf8");
    applyClientConfig("claude", SERVER_ENTRY, configPath);

    const json = JSON.parse(readFileSync(configPath, "utf8"));
    expect(json.theme).toBe("dark");
    expect(json.mcpServers?.parallax).toBeDefined();
    rmSync(tmp, { recursive: true });
  });

  it("AC03: gemini — creates settings.json with mcpServers.parallax", () => {
    const tmp = makeTempDir();
    const configPath = join(tmp, "settings.json");
    const result = applyClientConfig("gemini", SERVER_ENTRY, configPath);

    expect(result.success).toBe(true);
    const json = JSON.parse(readFileSync(configPath, "utf8"));
    expect(json.mcpServers?.parallax?.args).toEqual(SERVER_ENTRY.args);
    rmSync(tmp, { recursive: true });
  });

  it("AC04: github-copilot — creates mcp-config.json with mcpServers.parallax", () => {
    const tmp = makeTempDir();
    const configPath = join(tmp, "mcp-config.json");
    const result = applyClientConfig("github-copilot", SERVER_ENTRY, configPath);

    expect(result.success).toBe(true);
    const json = JSON.parse(readFileSync(configPath, "utf8"));
    expect(json.mcpServers?.parallax?.command).toBe("bun");
    rmSync(tmp, { recursive: true });
  });

  it("AC05: codex — creates mcp.json with mcpServers.parallax", () => {
    const tmp = makeTempDir();
    const configPath = join(tmp, "mcp.json");
    const result = applyClientConfig("codex", SERVER_ENTRY, configPath);

    expect(result.success).toBe(true);
    const json = JSON.parse(readFileSync(configPath, "utf8"));
    expect(json.mcpServers?.parallax?.command).toBe("bun");
    rmSync(tmp, { recursive: true });
  });

  it("AC05: codex — creates parent directory if missing", () => {
    const tmp = makeTempDir();
    const configPath = join(tmp, "nested", "deep", "mcp.json");
    const result = applyClientConfig("codex", SERVER_ENTRY, configPath);

    expect(result.success).toBe(true);
    expect(existsSync(configPath)).toBe(true);
    rmSync(tmp, { recursive: true });
  });

  it("AC05: returns success=false and error when write fails (bad path)", () => {
    // Pass a directory as configPath so writeFileSync will throw
    const tmp = makeTempDir();
    const result = applyClientConfig("codex", SERVER_ENTRY, tmp); // tmp is a dir, not a file
    expect(result.success).toBe(false);
    expect(result.error).toBeTruthy();
    rmSync(tmp, { recursive: true });
  });
});

// ── US-002 AC06: success message in CLI output ────────────────────────────────

describe("parallax mcp install — success message (US-002-AC06)", () => {
  it("AC06: output includes client name after successful config write", async () => {
    // No-TTY subprocess falls back to non-interactive; the temp HOME ensures no
    // filesystem conflicts with real user configs.
    const tmp = makeTempDir();
    const proc = Bun.spawn(
      ["bun", "run", join(import.meta.dir, "../../src/index.ts"), "mcp", "install", "--non-interactive", "--clients", "gemini"],
      { stdout: "pipe", stderr: "pipe", env: { ...process.env, HOME: tmp } },
    );
    const [stdout, , exitCode] = await Promise.all([
      new Response(proc.stdout).text(),
      new Response(proc.stderr).text(),
      proc.exited,
    ]);
    expect(exitCode).toBe(0);
    expect(stdout).toContain("Gemini");
    rmSync(tmp, { recursive: true });
  });

  it("AC06: output includes config path after successful write", async () => {
    const tmp = makeTempDir();
    const proc = Bun.spawn(
      ["bun", "run", join(import.meta.dir, "../../src/index.ts"), "mcp", "install", "--non-interactive", "--clients", "gemini"],
      { stdout: "pipe", stderr: "pipe", env: { ...process.env, HOME: tmp } },
    );
    const [stdout, , exitCode] = await Promise.all([
      new Response(proc.stdout).text(),
      new Response(proc.stderr).text(),
      proc.exited,
    ]);
    expect(exitCode).toBe(0);
    expect(stdout).toContain(".gemini");
    rmSync(tmp, { recursive: true });
  });
});

// ── US-002 AC07: getMcpServerEntry returns bun command ───────────────────────

describe("getMcpServerEntry (US-002)", () => {
  it("AC07: command is bun", () => {
    const entry = getMcpServerEntry();
    expect(entry.command).toBe("bun");
  });

  it("AC07: args contain parallax_mcp path", () => {
    const entry = getMcpServerEntry();
    expect(entry.args.some((a) => a.includes("parallax_mcp"))).toBe(true);
  });
});
