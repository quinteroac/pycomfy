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

describe("parallax CLI — create subcommand help (US-002)", () => {
  it("US-002-AC01: create --help prints usage for 'parallax create <media> [options]'", async () => {
    const { stdout, exitCode } = await runCLI(["create", "--help"]);
    expect(exitCode).toBe(0);
    expect(stdout).toContain("create");
    expect(stdout).toContain("<media>");
    expect(stdout).toContain("[options]");
  });

  it("US-002-AC02: create --help lists media types image, video, audio", async () => {
    const { stdout, exitCode } = await runCLI(["create", "--help"]);
    expect(exitCode).toBe(0);
    expect(stdout).toContain("image");
    expect(stdout).toContain("video");
    expect(stdout).toContain("audio");
  });
});

describe("parallax CLI — edit subcommand help (US-003)", () => {
  it("US-003-AC01: edit --help prints usage for 'parallax edit <media> [options]'", async () => {
    const { stdout, exitCode } = await runCLI(["edit", "--help"]);
    expect(exitCode).toBe(0);
    expect(stdout).toContain("edit");
    expect(stdout).toContain("<media>");
    expect(stdout).toContain("[options]");
  });

  it("US-003-AC02: edit --help lists media types image and video", async () => {
    const { stdout, exitCode } = await runCLI(["edit", "--help"]);
    expect(exitCode).toBe(0);
    expect(stdout).toContain("image");
    expect(stdout).toContain("video");
  });
});

describe("parallax CLI — media-level help (US-004)", () => {
  // create image
  it("US-004-AC01a: create image --help shows image-specific flags", async () => {
    const { stdout, exitCode } = await runCLI(["create", "image", "--help"]);
    expect(exitCode).toBe(0);
    expect(stdout).toContain("--model");
    expect(stdout).toContain("--prompt");
    expect(stdout).toContain("--negative-prompt");
    expect(stdout).toContain("--width");
    expect(stdout).toContain("--height");
    expect(stdout).toContain("--steps");
    expect(stdout).toContain("--cfg");
    expect(stdout).toContain("--seed");
    expect(stdout).toContain("--output");
  });

  it("US-004-AC02a: create image --help footer lists available models", async () => {
    const { stdout, exitCode } = await runCLI(["create", "image", "--help"]);
    expect(exitCode).toBe(0);
    expect(stdout).toContain("Available models:");
    expect(stdout).toContain("sdxl");
    expect(stdout).toContain("anima");
    expect(stdout).toContain("z_image");
    expect(stdout).toContain("flux_klein");
    expect(stdout).toContain("qwen");
  });

  // create video
  it("US-004-AC01b: create video --help shows video-specific flags", async () => {
    const { stdout, exitCode } = await runCLI(["create", "video", "--help"]);
    expect(exitCode).toBe(0);
    expect(stdout).toContain("--model");
    expect(stdout).toContain("--prompt");
    expect(stdout).toContain("--width");
    expect(stdout).toContain("--height");
    expect(stdout).toContain("--length");
    expect(stdout).toContain("--steps");
    expect(stdout).toContain("--cfg");
    expect(stdout).toContain("--seed");
    expect(stdout).toContain("--output");
  });

  it("US-004-AC02b: create video --help footer lists available models", async () => {
    const { stdout, exitCode } = await runCLI(["create", "video", "--help"]);
    expect(exitCode).toBe(0);
    expect(stdout).toContain("Available models:");
    expect(stdout).toContain("ltx2");
    expect(stdout).toContain("ltx23");
    expect(stdout).toContain("wan21");
    expect(stdout).toContain("wan22");
  });

  // create audio
  it("US-004-AC01c: create audio --help shows audio-specific flags", async () => {
    const { stdout, exitCode } = await runCLI(["create", "audio", "--help"]);
    expect(exitCode).toBe(0);
    expect(stdout).toContain("--model");
    expect(stdout).toContain("--prompt");
    expect(stdout).toContain("--length");
    expect(stdout).toContain("--steps");
    expect(stdout).toContain("--seed");
    expect(stdout).toContain("--output");
  });

  it("US-004-AC02c: create audio --help footer lists available models", async () => {
    const { stdout, exitCode } = await runCLI(["create", "audio", "--help"]);
    expect(exitCode).toBe(0);
    expect(stdout).toContain("Available models:");
    expect(stdout).toContain("ace_step");
  });

  // edit image
  it("US-004-AC01d: edit image --help shows image-edit-specific flags", async () => {
    const { stdout, exitCode } = await runCLI(["edit", "image", "--help"]);
    expect(exitCode).toBe(0);
    expect(stdout).toContain("--model");
    expect(stdout).toContain("--prompt");
    expect(stdout).toContain("--input");
    expect(stdout).toContain("--steps");
    expect(stdout).toContain("--cfg");
    expect(stdout).toContain("--seed");
    expect(stdout).toContain("--output");
  });

  it("US-004-AC02d: edit image --help footer lists available models", async () => {
    const { stdout, exitCode } = await runCLI(["edit", "image", "--help"]);
    expect(exitCode).toBe(0);
    expect(stdout).toContain("Available models:");
    expect(stdout).toContain("qwen");
  });

  // edit video
  it("US-004-AC01e: edit video --help shows video-edit-specific flags", async () => {
    const { stdout, exitCode } = await runCLI(["edit", "video", "--help"]);
    expect(exitCode).toBe(0);
    expect(stdout).toContain("--model");
    expect(stdout).toContain("--prompt");
    expect(stdout).toContain("--input");
    expect(stdout).toContain("--width");
    expect(stdout).toContain("--height");
    expect(stdout).toContain("--length");
    expect(stdout).toContain("--steps");
    expect(stdout).toContain("--cfg");
    expect(stdout).toContain("--seed");
    expect(stdout).toContain("--output");
  });

  it("US-004-AC02e: edit video --help footer lists available models", async () => {
    const { stdout, exitCode } = await runCLI(["edit", "video", "--help"]);
    expect(exitCode).toBe(0);
    expect(stdout).toContain("Available models:");
    expect(stdout).toContain("wan21");
    expect(stdout).toContain("wan22");
  });
});

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
