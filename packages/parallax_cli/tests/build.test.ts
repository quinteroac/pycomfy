import { describe, it, expect } from "bun:test";
import { join, resolve } from "path";
import { readFileSync, existsSync } from "fs";

const PACKAGE_ROOT = join(import.meta.dir, "..");
const REPO_ROOT = resolve(PACKAGE_ROOT, "../../");
const PACKAGE_JSON_PATH = join(PACKAGE_ROOT, "package.json");

const pkg = JSON.parse(readFileSync(PACKAGE_JSON_PATH, "utf-8"));

// ── AC01: build scripts copy runtime/ into dist/runtime/ ─────────────────────

describe("build scripts — runtime copy (US-005-AC01)", () => {
  it("build:linux copies runtime into dist/runtime", () => {
    expect(pkg.scripts["build:linux"]).toContain("cp -r runtime dist/runtime");
  });

  it("build:mac copies runtime into dist/runtime", () => {
    expect(pkg.scripts["build:mac"]).toContain("cp -r runtime dist/runtime");
  });

  it("build:win copies runtime into dist/runtime", () => {
    expect(pkg.scripts["build:win"]).toContain("cp -r runtime dist/runtime");
  });
});

// ── AC02/AC03: build scripts produce binary + runtime copy ───────────────────

describe("build scripts — script ordering (US-005-AC02, US-005-AC03)", () => {
  it("build:linux compiles binary before copying runtime", () => {
    const script = pkg.scripts["build:linux"] as string;
    const binaryIdx = script.indexOf("--outfile dist/parallax-linux");
    const copyIdx = script.indexOf("cp -r runtime dist/runtime");
    expect(binaryIdx).toBeGreaterThan(-1);
    expect(copyIdx).toBeGreaterThan(binaryIdx);
  });

  it("build:mac compiles binary before copying runtime", () => {
    const script = pkg.scripts["build:mac"] as string;
    const binaryIdx = script.indexOf("--outfile dist/parallax-macos");
    const copyIdx = script.indexOf("cp -r runtime dist/runtime");
    expect(binaryIdx).toBeGreaterThan(-1);
    expect(copyIdx).toBeGreaterThan(binaryIdx);
  });
});

// ── AC04: dist/runtime/ is covered by .gitignore ────────────────────────────

describe("gitignore coverage (US-005-AC04)", () => {
  it("root .gitignore ignores dist/ (covers dist/runtime/)", () => {
    const gitignorePath = join(REPO_ROOT, ".gitignore");
    expect(existsSync(gitignorePath)).toBe(true);
    const content = readFileSync(gitignorePath, "utf-8");
    // dist/ pattern at root level covers packages/parallax_cli/dist/runtime/
    expect(content).toMatch(/^dist\/$/m);
  });
});
