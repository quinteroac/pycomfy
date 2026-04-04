import { describe, it, expect } from "bun:test";
import { join } from "path";
import { readFileSync } from "fs";

const PACKAGE_ROOT = join(import.meta.dir, "..");
const PACKAGE_JSON_PATH = join(PACKAGE_ROOT, "package.json");
const pkg = JSON.parse(readFileSync(PACKAGE_JSON_PATH, "utf-8"));

// ── AC05: bunqueue is listed as a dependency ──────────────────────────────────

describe("queue dependency (US-002-AC05)", () => {
  it("bunqueue is listed in dependencies", () => {
    expect(pkg.dependencies).toBeDefined();
    expect(pkg.dependencies["bunqueue"]).toBeDefined();
  });
});

// ── AC01: getQueue is exported from queue.ts ──────────────────────────────────

describe("getQueue export (US-002-AC01)", () => {
  it("getQueue is exported from queue.ts", async () => {
    const mod = await import("../src/queue");
    expect(typeof mod.getQueue).toBe("function");
  });

  it("getQueue is re-exported from index.ts", async () => {
    const mod = await import("../src/index");
    expect(typeof (mod as any).getQueue).toBe("function");
  });
});

// ── AC02: lazy singleton ──────────────────────────────────────────────────────

describe("getQueue singleton (US-002-AC02)", () => {
  it("returns the same instance on subsequent calls", async () => {
    const { getQueue } = await import("../src/queue");
    const a = getQueue();
    const b = getQueue();
    expect(a).toBe(b);
  });
});

// ── AC03: database path resolves to ~/.config/parallax/jobs.db ───────────────

describe("database path (US-002-AC03)", () => {
  it("queue dataPath resolves to ~/.config/parallax/jobs.db", async () => {
    const src = readFileSync(join(PACKAGE_ROOT, "src", "queue.ts"), "utf-8");
    expect(src).toContain("os.homedir()");
    expect(src).toContain(".config");
    expect(src).toContain("parallax");
    expect(src).toContain("jobs.db");
  });
});

// ── AC04: embedded mode, no processor ────────────────────────────────────────

describe("queue creation options (US-002-AC04)", () => {
  it("queue is created with embedded: true", () => {
    const src = readFileSync(join(PACKAGE_ROOT, "src", "queue.ts"), "utf-8");
    expect(src).toContain("embedded: true");
  });

  it("no processor or routes are attached — consumers own the processors", () => {
    const src = readFileSync(join(PACKAGE_ROOT, "src", "queue.ts"), "utf-8");
    expect(src).not.toMatch(/routes\s*:/);
    expect(src).not.toMatch(/processor\s*:/);
  });
});
