import { describe, it, expect, beforeEach, afterEach } from "bun:test";
import { join } from "path";
import { mkdirSync, rmSync, readFileSync, writeFileSync, existsSync } from "fs";
import { homedir } from "os";

// Paths used by the module under test
const CONFIG_DIR = join(homedir(), ".config", "parallax");
const CONFIG_PATH = join(CONFIG_DIR, "config.json");

// Back up and restore config.json around tests to avoid polluting the real user config.
let backup: string | null = null;

beforeEach(() => {
  backup = existsSync(CONFIG_PATH) ? readFileSync(CONFIG_PATH, "utf-8") : null;
  // Remove config so each test starts clean
  if (existsSync(CONFIG_PATH)) rmSync(CONFIG_PATH);
  // Clear relevant env vars
  delete process.env.PARALLAX_REPO_ROOT;
  delete process.env.PYCOMFY_MODELS_DIR;
});

afterEach(() => {
  // Restore original config
  if (backup !== null) {
    if (!existsSync(CONFIG_DIR)) mkdirSync(CONFIG_DIR, { recursive: true });
    writeFileSync(CONFIG_PATH, backup, "utf-8");
  } else if (existsSync(CONFIG_PATH)) {
    rmSync(CONFIG_PATH);
  }
  delete process.env.PARALLAX_REPO_ROOT;
  delete process.env.PYCOMFY_MODELS_DIR;
});

// Import after env setup so module re-reads env each call
import { readConfig, writeConfig, configExists } from "./config";

// US-002-AC01: exports the three required functions
describe("config exports", () => {
  it("exports readConfig, writeConfig, configExists", () => {
    expect(typeof readConfig).toBe("function");
    expect(typeof writeConfig).toBe("function");
    expect(typeof configExists).toBe("function");
  });
});

// US-002-AC02: stored at correct path with correct fields
describe("writeConfig / configExists", () => {
  it("configExists returns false when no config file", () => {
    expect(configExists()).toBe(false);
  });

  it("writeConfig creates config.json with all fields", () => {
    const now = new Date().toISOString();
    writeConfig({ repoRoot: "/repo", modelsDir: "/models", uvPath: "/usr/bin/uv", installedAt: now });

    expect(configExists()).toBe(true);
    const raw = JSON.parse(readFileSync(CONFIG_PATH, "utf-8"));
    expect(raw.repoRoot).toBe("/repo");
    expect(raw.modelsDir).toBe("/models");
    expect(raw.uvPath).toBe("/usr/bin/uv");
    expect(raw.installedAt).toBe(now);
  });

  it("writeConfig creates parent directories if missing", () => {
    if (existsSync(CONFIG_DIR)) rmSync(CONFIG_DIR, { recursive: true });
    writeConfig({ repoRoot: "/r" });
    expect(existsSync(CONFIG_PATH)).toBe(true);
  });
});

// US-002-AC03: env vars override stored values
describe("readConfig merges env vars", () => {
  it("returns empty object when no config and no env vars", () => {
    const cfg = readConfig();
    expect(cfg.repoRoot).toBeUndefined();
    expect(cfg.modelsDir).toBeUndefined();
  });

  it("returns stored values when no env vars set", () => {
    writeConfig({ repoRoot: "/stored-repo", modelsDir: "/stored-models" });
    const cfg = readConfig();
    expect(cfg.repoRoot).toBe("/stored-repo");
    expect(cfg.modelsDir).toBe("/stored-models");
  });

  it("PARALLAX_REPO_ROOT env var overrides stored repoRoot", () => {
    writeConfig({ repoRoot: "/stored-repo" });
    process.env.PARALLAX_REPO_ROOT = "/env-repo";
    const cfg = readConfig();
    expect(cfg.repoRoot).toBe("/env-repo");
  });

  it("PYCOMFY_MODELS_DIR env var overrides stored modelsDir", () => {
    writeConfig({ modelsDir: "/stored-models" });
    process.env.PYCOMFY_MODELS_DIR = "/env-models";
    const cfg = readConfig();
    expect(cfg.modelsDir).toBe("/env-models");
  });

  it("both env vars override stored values simultaneously", () => {
    writeConfig({ repoRoot: "/stored-repo", modelsDir: "/stored-models" });
    process.env.PARALLAX_REPO_ROOT = "/env-repo";
    process.env.PYCOMFY_MODELS_DIR = "/env-models";
    const cfg = readConfig();
    expect(cfg.repoRoot).toBe("/env-repo");
    expect(cfg.modelsDir).toBe("/env-models");
  });

  it("non-overridden stored fields are preserved when env vars are set", () => {
    const now = new Date().toISOString();
    writeConfig({ repoRoot: "/stored-repo", uvPath: "/uv", installedAt: now });
    process.env.PARALLAX_REPO_ROOT = "/env-repo";
    const cfg = readConfig();
    expect(cfg.repoRoot).toBe("/env-repo");
    expect(cfg.uvPath).toBe("/uv");
    expect(cfg.installedAt).toBe(now);
  });
});
