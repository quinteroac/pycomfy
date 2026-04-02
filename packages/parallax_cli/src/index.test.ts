import { describe, it, expect } from "bun:test";
import { join } from "path";

const CLI = join(import.meta.dir, "index.ts");

async function runCLI(args: string[]): Promise<{ stdout: string; stderr: string; exitCode: number }> {
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

describe("parallax CLI — top-level help (US-001)", () => {
  it("US-001-AC01: --help prints tool name, version, description, and subcommands", async () => {
    const { stdout, exitCode } = await runCLI(["--help"]);
    expect(exitCode).toBe(0);
    expect(stdout).toContain("parallax");
    expect(stdout).toContain("0.1.0");
    expect(stdout).toContain("Parallax CLI");
    expect(stdout).toContain("create");
    expect(stdout).toContain("edit");
  });

  it("US-001-AC02: no arguments shows help and exits with code 0", async () => {
    const { stdout, exitCode } = await runCLI([]);
    expect(exitCode).toBe(0);
    expect(stdout).toContain("parallax");
    expect(stdout).toContain("create");
    expect(stdout).toContain("edit");
  });
});
