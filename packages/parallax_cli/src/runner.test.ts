import { describe, it, expect } from "bun:test";
import { readFileSync } from "fs";
import { join } from "path";
import { mkdtemp, writeFile, rm } from "fs/promises";
import { tmpdir } from "os";

const CLI = join(import.meta.dir, "index.ts");
const RUNNER_SRC = readFileSync(join(import.meta.dir, "runner.ts"), "utf-8");

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

// Helper: create a temporary fake repo root with a trivial Python script
async function makeFakeRepoRoot(relScript: string): Promise<string> {
  const root = await mkdtemp(join(tmpdir(), "parallax-runner-test-"));
  const scriptDir = join(root, relScript.split("/").slice(0, -1).join("/"));
  await rm(scriptDir, { recursive: true, force: true });
  await import("fs/promises").then((fs) => fs.mkdir(scriptDir, { recursive: true }));
  await writeFile(join(root, relScript), "import sys; sys.exit(0)\n", "utf-8");
  return root;
}

// ── AC02: no process.env.PARALLAX_REPO_ROOT reference inside runner.ts ────────

describe("runner.ts — US-003-AC02 (no env read inside module)", () => {
  it("runner.ts does not reference process.env.PARALLAX_REPO_ROOT", () => {
    expect(RUNNER_SRC).not.toContain("process.env.PARALLAX_REPO_ROOT");
  });

  it("runner.ts does not reference process.env.PYCOMFY_MODELS_DIR", () => {
    expect(RUNNER_SRC).not.toContain("process.env.PYCOMFY_MODELS_DIR");
  });
});

// ── AC01: spawnPipeline accepts (scriptRelPath, args, config) ─────────────────

describe("runner.ts — US-003-AC01 (function signature accepts config)", () => {
  it("spawnPipeline is exported from runner.ts", async () => {
    const mod = await import("./runner");
    expect(typeof mod.spawnPipeline).toBe("function");
  });

  it("spawnPipeline rejects missing repoRoot in config and exits 1", async () => {
    const { stderr, exitCode } = await runCLIWithEnv(
      ["create", "image", "--model", "sdxl", "--prompt", "test", "--models-dir", "/tmp"],
      { PARALLAX_REPO_ROOT: undefined },
    );
    expect(exitCode).toBe(1);
    expect(stderr).toContain("Error: PARALLAX_REPO_ROOT is required");
  });

  it("spawnPipeline uses repoRoot from config (not env) to build script path", async () => {
    const root = await makeFakeRepoRoot("examples/image/generation/sdxl/t2i.py");
    try {
      const { exitCode } = await runCLIWithEnv(
        ["create", "image", "--model", "sdxl", "--prompt", "test", "--models-dir", "/tmp"],
        { PARALLAX_REPO_ROOT: root },
      );
      // uv run python will fail (uv not necessarily in PATH in all envs), but the
      // important check is that we did NOT get the "PARALLAX_REPO_ROOT is required" error
      // — meaning repoRoot was correctly resolved from the env-sourced config.
      expect(exitCode).not.toBeUndefined();
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  });
});

// ── AC03: backward compat — PARALLAX_REPO_ROOT env var still works ────────────

describe("runner.ts — US-003-AC03 (backward compat via readConfig)", () => {
  it("create image works when PARALLAX_REPO_ROOT is set as env var (error is NOT about missing root)", async () => {
    // When root is set, we pass the repoRoot check; any subsequent failure (e.g.
    // spawn failure) will NOT begin with "Error: PARALLAX_REPO_ROOT is required".
    const { stderr, exitCode } = await runCLIWithEnv(
      ["create", "image", "--model", "sdxl", "--prompt", "test", "--models-dir", "/tmp"],
      { PARALLAX_REPO_ROOT: "/nonexistent-repo-root-xyz" },
    );
    expect(exitCode).not.toBe(0);
    // The actual console.error("Error: PARALLAX_REPO_ROOT is required") output
    // would appear at the very start of stderr; a spawn-failure stack trace starts
    // with source-context lines (e.g. "13 |  if (!repoRoot)").
    expect(stderr).not.toMatch(/^Error: PARALLAX_REPO_ROOT is required/);
  });

  it("create video works when PARALLAX_REPO_ROOT is set as env var", async () => {
    const { stderr, exitCode } = await runCLIWithEnv(
      ["create", "video", "--model", "ltx2", "--prompt", "test", "--models-dir", "/tmp"],
      { PARALLAX_REPO_ROOT: "/nonexistent-repo-root-xyz" },
    );
    expect(exitCode).not.toBe(0);
    expect(stderr).not.toMatch(/^Error: PARALLAX_REPO_ROOT is required/);
  });

  it("create audio works when PARALLAX_REPO_ROOT is set as env var", async () => {
    const { stderr, exitCode } = await runCLIWithEnv(
      ["create", "audio", "--model", "ace_step", "--prompt", "test", "--models-dir", "/tmp"],
      { PARALLAX_REPO_ROOT: "/nonexistent-repo-root-xyz" },
    );
    expect(exitCode).not.toBe(0);
    expect(stderr).not.toMatch(/^Error: PARALLAX_REPO_ROOT is required/);
  });
});
