import { describe, it, expect, beforeEach, afterEach } from "bun:test";
import { join } from "path";
import { existsSync, readFileSync, unlinkSync, mkdirSync, readdirSync, rmSync } from "fs";
import { homedir } from "os";

const CLI = join(import.meta.dir, "../../src/index.ts");
const CONFIG_PATH = join(homedir(), ".config", "parallax", "config.json");
const INSTALLED_RUNTIME_DIR = join(homedir(), ".config", "parallax", "runtime");

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

// Back up and restore the real config around each test so we don't corrupt developer state.
let backupConfig: string | null = null;

beforeEach(() => {
  if (existsSync(CONFIG_PATH)) {
    backupConfig = readFileSync(CONFIG_PATH, "utf-8");
  } else {
    backupConfig = null;
    // Ensure the directory exists so writeConfig works in the test subject.
    mkdirSync(join(homedir(), ".config", "parallax"), { recursive: true });
  }
});

afterEach(() => {
  if (backupConfig !== null) {
    import("fs").then(({ writeFileSync }) =>
      writeFileSync(CONFIG_PATH, backupConfig as string, "utf-8"),
    );
  } else if (existsSync(CONFIG_PATH)) {
    unlinkSync(CONFIG_PATH);
  }
});

// ── AC01: command registered ──────────────────────────────────────────────────

describe("parallax install — command registration (US-007-AC01)", () => {
  it("install --help shows the install command description", async () => {
    const { stdout, exitCode } = await runCLI(["install", "--help"]);
    expect(exitCode).toBe(0);
    expect(stdout).toContain("install");
  });

  it("install --help lists --non-interactive flag", async () => {
    const { stdout, exitCode } = await runCLI(["install", "--help"]);
    expect(exitCode).toBe(0);
    expect(stdout).toContain("--non-interactive");
  });

  it("install --help lists --install-dir flag", async () => {
    const { stdout, exitCode } = await runCLI(["install", "--help"]);
    expect(exitCode).toBe(0);
    expect(stdout).toContain("--install-dir");
  });

  it("install --help lists --models-dir flag", async () => {
    const { stdout, exitCode } = await runCLI(["install", "--help"]);
    expect(exitCode).toBe(0);
    expect(stdout).toContain("--models-dir");
  });

  it("install --help lists --variant flag", async () => {
    const { stdout, exitCode } = await runCLI(["install", "--help"]);
    expect(exitCode).toBe(0);
    expect(stdout).toContain("--variant");
  });
});

// ── AC03: non-interactive mode accepts all flags ──────────────────────────────

describe("parallax install --non-interactive (US-007-AC03/AC05)", () => {
  it("AC03: accepts --install-dir, --models-dir, --variant cuda and exits 0", async () => {
    const { stdout, exitCode } = await runCLI([
      "install",
      "--non-interactive",
      "--install-dir",
      "/tmp/parallax-test",
      "--models-dir",
      "/tmp/parallax-models-test",
      "--variant",
      "cuda",
    ]);
    expect(exitCode).toBe(0);
    expect(stdout).toContain("install-dir: /tmp/parallax-test");
    expect(stdout).toContain("models-dir:  /tmp/parallax-models-test");
    expect(stdout).toContain("variant:     cuda");
  });

  it("AC03: accepts --variant cpu", async () => {
    const { stdout, exitCode } = await runCLI([
      "install",
      "--non-interactive",
      "--install-dir",
      "/tmp/parallax-test",
      "--models-dir",
      "/tmp/parallax-models-test",
      "--variant",
      "cpu",
    ]);
    expect(exitCode).toBe(0);
    expect(stdout).toContain("variant:     cpu");
  });

  it("AC05: --non-interactive with no extra flags runs to completion without crash", async () => {
    const { exitCode } = await runCLI(["install", "--non-interactive"]);
    expect(exitCode).toBe(0);
  });

  it("AC05: writes config to ~/.config/parallax/config.json", async () => {
    await runCLI([
      "install",
      "--non-interactive",
      "--install-dir",
      "/tmp/install-test-dir",
      "--models-dir",
      "/tmp/install-test-models",
      "--variant",
      "cpu",
    ]);
    expect(existsSync(CONFIG_PATH)).toBe(true);
    const written = JSON.parse(readFileSync(CONFIG_PATH, "utf-8"));
    expect(written.repoRoot).toBe("/tmp/install-test-dir");
    expect(written.modelsDir).toBe("/tmp/install-test-models");
    expect(written.variant).toBe("cpu");
    expect(typeof written.installedAt).toBe("string");
  });

  it("AC05: confirmation message printed", async () => {
    const { stdout } = await runCLI(["install", "--non-interactive"]);
    expect(stdout).toContain("[parallax] Installation configured");
    expect(stdout).toContain("Configuration saved");
  });
});

// ── AC04: auto-fallback when no TTY ──────────────────────────────────────────
// When stdout is a pipe (not a TTY), the command falls back to non-interactive.
// Bun.spawn with stdout:"pipe" means stdout is not a TTY inside the subprocess.

describe("parallax install — auto non-interactive fallback (US-007-AC04)", () => {
  it("AC04: runs without crash when stdout is not a TTY (piped subprocess)", async () => {
    // In this test setup stdout is already piped (not a TTY), so this exercises AC04.
    const { stdout, exitCode } = await runCLI([
      "install",
      "--install-dir",
      "/tmp/parallax-test-notty",
      "--models-dir",
      "/tmp/parallax-models-notty",
      "--variant",
      "cpu",
    ]);
    expect(exitCode).toBe(0);
    expect(stdout).toContain("[parallax] Installation configured");
  });
});

// ── US-004: runtime/ copy ─────────────────────────────────────────────────────

describe("parallax install — copies runtime/ (US-004)", () => {
  let runtimeBackup: boolean;

  beforeEach(() => {
    runtimeBackup = existsSync(INSTALLED_RUNTIME_DIR);
  });

  afterEach(() => {
    // Clean up installed runtime dir only if it didn't exist before the test.
    if (!runtimeBackup && existsSync(INSTALLED_RUNTIME_DIR)) {
      rmSync(INSTALLED_RUNTIME_DIR, { recursive: true, force: true });
    }
  });

  it("AC01/AC02: --non-interactive copies runtime/ to ~/.config/parallax/runtime/ (idempotent)", async () => {
    const { exitCode } = await runCLI(["install", "--non-interactive"]);
    expect(exitCode).toBe(0);
    expect(existsSync(INSTALLED_RUNTIME_DIR)).toBe(true);
    // Should be a directory containing at least one subdirectory.
    const entries = readdirSync(INSTALLED_RUNTIME_DIR);
    expect(entries.length).toBeGreaterThan(0);

    // Idempotent: run again, should not fail.
    const { exitCode: exitCode2 } = await runCLI(["install", "--non-interactive"]);
    expect(exitCode2).toBe(0);
  });

  it("AC03: config stores runtimeDir pointing to ~/.config/parallax/runtime/", async () => {
    await runCLI(["install", "--non-interactive"]);
    const written = JSON.parse(readFileSync(CONFIG_PATH, "utf-8"));
    expect(written.runtimeDir).toBe(INSTALLED_RUNTIME_DIR);
  });

  it("AC04: interactive spinner logs 'Copying runtime' during non-interactive (no-TTY) run", async () => {
    const { stdout, exitCode } = await runCLI(["install", "--non-interactive"]);
    expect(exitCode).toBe(0);
    expect(stdout).toContain("Copying runtime");
  });

  it("AC05: non-interactive path logs runtime-dir to stdout", async () => {
    const { stdout, exitCode } = await runCLI(["install", "--non-interactive"]);
    expect(exitCode).toBe(0);
    expect(stdout).toContain("runtime-dir:");
    expect(stdout).toContain(INSTALLED_RUNTIME_DIR);
  });
});
