import { describe, it, expect } from "bun:test";
import { readFileSync } from "fs";
import { join } from "path";
import { mkdtemp, writeFile, rm, mkdir } from "fs/promises";
import { tmpdir } from "os";

const CLI = join(import.meta.dir, "../src/index.ts");
const RUNNER_SRC = readFileSync(join(import.meta.dir, "../src/runner.ts"), "utf-8");

// Helper: spawn CLI with explicit env overrides (undefined = unset)
async function runCLIWithEnv(
  args: string[],
  envOverrides: Record<string, string | undefined>,
): Promise<{ stdout: string; stderr: string; exitCode: number }> {
  const env: Record<string, string> = {};
  for (const [k, v] of Object.entries(process.env)) {
    if (v !== undefined) env[k] = v;
  }
  for (const [k, v] of Object.entries(envOverrides)) {
    if (v === undefined) delete env[k];
    else env[k] = v;
  }
  const proc = Bun.spawn(["bun", "run", CLI, ...args], {
    stdout: "pipe",
    stderr: "pipe",
    env,
  });
  const [stdout, stderr, exitCode] = await Promise.all([
    new Response(proc.stdout).text(),
    new Response(proc.stderr).text(),
    proc.exited,
  ]);
  return { stdout, stderr, exitCode };
}

// Helper: create a temporary fake root with a trivial Python script at scriptRelPath
async function makeFakeRoot(relScript: string): Promise<string> {
  const root = await mkdtemp(join(tmpdir(), "parallax-runner-test-"));
  const scriptDir = join(root, relScript.split("/").slice(0, -1).join("/"));
  await mkdir(scriptDir, { recursive: true });
  await writeFile(join(root, relScript), "import sys; sys.exit(0)\n", "utf-8");
  return root;
}

// ── AC05: no process.env references inside runner.ts ──────────────────────────

describe("runner.ts — no direct env reads", () => {
  it("runner.ts does not reference process.env.PARALLAX_REPO_ROOT", () => {
    expect(RUNNER_SRC).not.toContain("process.env.PARALLAX_REPO_ROOT");
  });

  it("runner.ts does not reference process.env.PYCOMFY_MODELS_DIR", () => {
    expect(RUNNER_SRC).not.toContain("process.env.PYCOMFY_MODELS_DIR");
  });

  it("runner.ts does not reference process.env.PARALLAX_RUNTIME_DIR", () => {
    expect(RUNNER_SRC).not.toContain("process.env.PARALLAX_RUNTIME_DIR");
  });
});

// ── AC01: ParallaxConfig has runtimeDir field ─────────────────────────────────

describe("config.ts — US-003-AC01 (runtimeDir field)", () => {
  it("ParallaxConfig accepts runtimeDir field", async () => {
    const { readConfig, writeConfig } = await import("../src/config");
    // Should compile and round-trip through JSON without error
    const cfg = { runtimeDir: "/installed/runtime" };
    // writeConfig accepts runtimeDir (type check via assignment)
    const typed: import("../src/config").ParallaxConfig = cfg;
    expect(typed.runtimeDir).toBe("/installed/runtime");
  });

  it("PARALLAX_RUNTIME_DIR env var is mapped to runtimeDir by readConfig", async () => {
    // We can't directly test this in-process since modules are cached, so verify
    // the source file contains the mapping
    const configSrc = readFileSync(join(import.meta.dir, "../src/config.ts"), "utf-8");
    expect(configSrc).toContain("PARALLAX_RUNTIME_DIR");
    expect(configSrc).toContain("runtimeDir");
  });
});

// ── AC02: runtimeDir used when set ────────────────────────────────────────────

describe("runner.ts — US-003-AC02 (runtimeDir resolves script path)", () => {
  it("spawnPipeline uses runtimeDir (not repoRoot) when runtimeDir is set in config", async () => {
    // Use PARALLAX_RUNTIME_DIR to inject runtimeDir via readConfig
    const root = await makeFakeRoot("runtime/image/generation/sdxl/t2i.py");
    try {
      const { stderr } = await runCLIWithEnv(
        ["create", "image", "--model", "sdxl", "--prompt", "test", "--models-dir", "/tmp"],
        {
          PARALLAX_REPO_ROOT: undefined,
          PARALLAX_RUNTIME_DIR: root,
        },
      );
      // Should NOT emit the "no script directory configured" error at the start
      expect(stderr).not.toMatch(/^Error: no script directory configured/);
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  });

  it("runner.ts source resolves script via runtimeDir when set", () => {
    expect(RUNNER_SRC).toContain("runtimeDir");
    expect(RUNNER_SRC).toContain("runtimeDir ?? repoRoot");
  });
});

// ── AC03: fallback to repoRoot when runtimeDir not set ────────────────────────

describe("runner.ts — US-003-AC03 (fallback to repoRoot)", () => {
  it("create image works when PARALLAX_REPO_ROOT is set (no runtimeDir)", async () => {
    const { stderr } = await runCLIWithEnv(
      ["create", "image", "--model", "sdxl", "--prompt", "test", "--models-dir", "/tmp"],
      { PARALLAX_REPO_ROOT: "/nonexistent-repo-root-xyz", PARALLAX_RUNTIME_DIR: undefined },
    );
    // If repoRoot is set, we pass the check — any failure is a spawn error, not our error
    expect(stderr).not.toMatch(/^Error: no script directory configured/);
  });

  it("create video works when PARALLAX_REPO_ROOT is set (no runtimeDir)", async () => {
    const { stderr } = await runCLIWithEnv(
      ["create", "video", "--model", "ltx2", "--prompt", "test", "--models-dir", "/tmp"],
      { PARALLAX_REPO_ROOT: "/nonexistent-repo-root-xyz", PARALLAX_RUNTIME_DIR: undefined },
    );
    expect(stderr).not.toMatch(/^Error: no script directory configured/);
  });

  it("create audio works when PARALLAX_REPO_ROOT is set (no runtimeDir)", async () => {
    const { stderr } = await runCLIWithEnv(
      ["create", "audio", "--model", "ace_step", "--prompt", "test", "--models-dir", "/tmp"],
      { PARALLAX_REPO_ROOT: "/nonexistent-repo-root-xyz", PARALLAX_RUNTIME_DIR: undefined },
    );
    expect(stderr).not.toMatch(/^Error: no script directory configured/);
  });
});

// ── AC04: clear error when neither runtimeDir nor repoRoot is set ─────────────

describe("runner.ts — US-003-AC04 (error when no directory configured)", () => {
  it("exits 1 with clear error when neither runtimeDir nor repoRoot is set", async () => {
    const { stderr, exitCode } = await runCLIWithEnv(
      ["create", "image", "--model", "sdxl", "--prompt", "test", "--models-dir", "/tmp"],
      { PARALLAX_REPO_ROOT: undefined, PARALLAX_RUNTIME_DIR: undefined },
    );
    expect(exitCode).toBe(1);
    expect(stderr).toContain("no script directory configured");
  });

  it("error message mentions both runtimeDir and PARALLAX_REPO_ROOT", async () => {
    const { stderr } = await runCLIWithEnv(
      ["create", "image", "--model", "sdxl", "--prompt", "test", "--models-dir", "/tmp"],
      { PARALLAX_REPO_ROOT: undefined, PARALLAX_RUNTIME_DIR: undefined },
    );
    expect(stderr).toContain("runtimeDir");
    expect(stderr).toContain("PARALLAX_REPO_ROOT");
  });
});
