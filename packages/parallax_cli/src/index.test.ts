import { describe, it, expect } from "bun:test";
import { join } from "path";
import { mkdtemp, mkdir, writeFile, rm } from "fs/promises";
import { tmpdir } from "os";

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

// Spawn CLI with explicit env overrides (undefined value = unset the key).
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

describe("parallax CLI — known-model validation (US-005)", () => {
  it("US-005-AC01/02: create image with unknown model prints error message and exits 1", async () => {
    const { stderr, exitCode } = await runCLI(["create", "image", "--model", "badmodel", "--prompt", "test"]);
    expect(exitCode).toBe(1);
    expect(stderr).toContain('unknown model "badmodel" for create image');
    expect(stderr).toContain("Known models:");
    expect(stderr).toContain("sdxl");
  });

  it("US-005-AC01/02: create video with unknown model prints error message and exits 1", async () => {
    const { stderr, exitCode } = await runCLI(["create", "video", "--model", "badmodel", "--prompt", "test"]);
    expect(exitCode).toBe(1);
    expect(stderr).toContain('unknown model "badmodel" for create video');
    expect(stderr).toContain("Known models:");
    expect(stderr).toContain("ltx2");
  });

  it("US-005-AC01/02: create audio with unknown model prints error message and exits 1", async () => {
    const { stderr, exitCode } = await runCLI(["create", "audio", "--model", "badmodel", "--prompt", "test"]);
    expect(exitCode).toBe(1);
    expect(stderr).toContain('unknown model "badmodel" for create audio');
    expect(stderr).toContain("Known models:");
    expect(stderr).toContain("ace_step");
  });

  it("US-005-AC01/02: edit image with unknown model prints error message and exits 1", async () => {
    const { stderr, exitCode } = await runCLI(["edit", "image", "--model", "badmodel", "--prompt", "test", "--input", "/etc/hostname"]);
    expect(exitCode).toBe(1);
    expect(stderr).toContain('unknown model "badmodel" for edit image');
    expect(stderr).toContain("Known models:");
    expect(stderr).toContain("qwen");
  });

  it("US-005-AC01/02: edit video with unknown model prints error message and exits 1", async () => {
    const { stderr, exitCode } = await runCLI(["edit", "video", "--model", "badmodel", "--prompt", "test", "--input", "vid.mp4"]);
    expect(exitCode).toBe(1);
    expect(stderr).toContain('unknown model "badmodel" for edit video');
    expect(stderr).toContain("Known models:");
    expect(stderr).toContain("wan21");
  });

  it("US-005: known model passes validation (no model-error in stderr)", async () => {
    const { stderr } = await runCLI(["create", "image", "--model", "sdxl", "--prompt", "test"]);
    expect(stderr).not.toContain("unknown model");
  });
});

describe("parallax CLI — required-flag validation (US-006)", () => {
  // AC01: omitting --model on create/edit commands
  it("US-006-AC01: create image without --model prints 'Error: --model is required' and exits 1", async () => {
    const { stderr, exitCode } = await runCLI(["create", "image", "--prompt", "test"]);
    expect(exitCode).toBe(1);
    expect(stderr).toContain("Error: --model is required");
  });

  it("US-006-AC01: create video without --model prints 'Error: --model is required' and exits 1", async () => {
    const { stderr, exitCode } = await runCLI(["create", "video", "--prompt", "test"]);
    expect(exitCode).toBe(1);
    expect(stderr).toContain("Error: --model is required");
  });

  it("US-006-AC01: create audio without --model prints 'Error: --model is required' and exits 1", async () => {
    const { stderr, exitCode } = await runCLI(["create", "audio", "--prompt", "test"]);
    expect(exitCode).toBe(1);
    expect(stderr).toContain("Error: --model is required");
  });

  it("US-006-AC01: edit image without --model prints 'Error: --model is required' and exits 1", async () => {
    const { stderr, exitCode } = await runCLI(["edit", "image", "--prompt", "test", "--input", "/etc/hostname"]);
    expect(exitCode).toBe(1);
    expect(stderr).toContain("Error: --model is required");
  });

  it("US-006-AC01: edit video without --model prints 'Error: --model is required' and exits 1", async () => {
    const { stderr, exitCode } = await runCLI(["edit", "video", "--prompt", "test", "--input", "vid.mp4"]);
    expect(exitCode).toBe(1);
    expect(stderr).toContain("Error: --model is required");
  });

  // AC01: omitting --prompt on create/edit commands
  it("US-006-AC01: create image without --prompt prints 'Error: --prompt is required' and exits 1", async () => {
    const { stderr, exitCode } = await runCLI(["create", "image", "--model", "sdxl"]);
    expect(exitCode).toBe(1);
    expect(stderr).toContain("Error: --prompt is required");
  });

  it("US-006-AC01: edit image without --prompt prints 'Error: --prompt is required' and exits 1", async () => {
    const { stderr, exitCode } = await runCLI(["edit", "image", "--model", "qwen", "--input", "/etc/hostname"]);
    expect(exitCode).toBe(1);
    expect(stderr).toContain("Error: --prompt is required");
  });

  // AC02: omitting --input on edit commands
  it("US-006-AC02: edit image without --input prints 'Error: --input is required' and exits 1", async () => {
    const { stderr, exitCode } = await runCLI(["edit", "image", "--model", "qwen", "--prompt", "test"]);
    expect(exitCode).toBe(1);
    expect(stderr).toContain("Error: --input is required");
  });

  it("US-006-AC02: edit video without --input prints 'Error: --input is required' and exits 1", async () => {
    const { stderr, exitCode } = await runCLI(["edit", "video", "--model", "wan21", "--prompt", "test"]);
    expect(exitCode).toBe(1);
    expect(stderr).toContain("Error: --input is required");
  });

  // AC03: exit code 1
  it("US-006-AC03: process exits with code 1 when required flag is missing", async () => {
    const { exitCode } = await runCLI(["create", "image"]);
    expect(exitCode).toBe(1);
  });
});

describe("parallax CLI — stub execution (US-007)", () => {
  it("US-007-AC01/02: edit video with unimplemented model prints stub message and exits 0", async () => {
    const { stdout, exitCode } = await runCLI(["edit", "video", "--model", "wan22", "--prompt", "test", "--input", "vid.mp4"]);
    expect(exitCode).toBe(0);
    expect(stdout).toContain("[parallax] edit video --model wan22 — not yet implemented (coming soon)");
  });

  it("US-007-AC01/02: create audio with valid flags prints stub message and exits 0", async () => {
    const { stdout, exitCode } = await runCLI(["create", "audio", "--model", "ace_step", "--prompt", "test"]);
    expect(exitCode).toBe(0);
    expect(stdout).toContain("[parallax] create audio --model ace_step — not yet implemented (coming soon)");
  });

  it("US-007-AC01/02: edit image with valid flags prints stub message and exits 0", async () => {
    const { stdout, exitCode } = await runCLI(["edit", "image", "--model", "qwen", "--prompt", "test", "--input", "/etc/hostname"]);
    expect(exitCode).toBe(0);
    expect(stdout).toContain("[parallax] edit image --model qwen — not yet implemented (coming soon)");
  });

  it("US-007-AC01/02: edit video with valid flags prints stub message and exits 0", async () => {
    const { stdout, exitCode } = await runCLI(["edit", "video", "--model", "wan21", "--prompt", "test", "--input", "vid.mp4"]);
    expect(exitCode).toBe(0);
    expect(stdout).toContain("[parallax] edit video --model wan21 — not yet implemented (coming soon)");
  });

  it("US-007: create image --model flux_klein still prints notImplemented (not yet wired)", async () => {
    const { stdout, exitCode } = await runCLIWithEnv(
      ["create", "image", "--model", "flux_klein", "--prompt", "test", "--models-dir", "/tmp"],
      { PARALLAX_REPO_ROOT: "/tmp" },
    );
    expect(exitCode).toBe(0);
    expect(stdout).toContain("[parallax] create image --model flux_klein — not yet implemented (coming soon)");
  });
});

describe("parallax CLI — sdxl image generation (US-001-it39)", () => {
  it("US-001-AC04: missing PYCOMFY_MODELS_DIR and --models-dir exits 1 with clear error", async () => {
    const { stderr, exitCode } = await runCLIWithEnv(
      ["create", "image", "--model", "sdxl", "--prompt", "test"],
      { PYCOMFY_MODELS_DIR: undefined, PARALLAX_REPO_ROOT: undefined },
    );
    expect(exitCode).toBe(1);
    expect(stderr).toContain("Error: --models-dir or PYCOMFY_MODELS_DIR is required");
  });

  it("US-001-AC04: --models-dir flag takes precedence; missing PARALLAX_REPO_ROOT exits 1", async () => {
    const { stderr, exitCode } = await runCLIWithEnv(
      ["create", "image", "--model", "sdxl", "--prompt", "test", "--models-dir", "/tmp"],
      { PARALLAX_REPO_ROOT: undefined },
    );
    expect(exitCode).toBe(1);
    expect(stderr).toContain("Error: PARALLAX_REPO_ROOT is required");
  });

  it("US-001-AC02: all optional flags are forwarded (no error about flags before PARALLAX_REPO_ROOT check)", async () => {
    const { stderr, exitCode } = await runCLIWithEnv(
      [
        "create", "image", "--model", "sdxl", "--prompt", "test",
        "--models-dir", "/tmp", "--negative-prompt", "bad quality",
        "--width", "512", "--height", "512", "--steps", "10", "--cfg", "5", "--seed", "42",
        "--output", "out.png",
      ],
      { PARALLAX_REPO_ROOT: undefined },
    );
    expect(exitCode).toBe(1);
    expect(stderr).toContain("Error: PARALLAX_REPO_ROOT is required");
  });

  it("US-001-AC03: CLI exits with non-zero when subprocess fails (bad PARALLAX_REPO_ROOT path)", async () => {
    const { exitCode } = await runCLIWithEnv(
      ["create", "image", "--model", "sdxl", "--prompt", "test", "--models-dir", "/tmp"],
      { PARALLAX_REPO_ROOT: "/nonexistent-parallax-root-12345" },
    );
    expect(exitCode).not.toBe(0);
  });

  it("US-001: --models-dir option is listed in create image --help", async () => {
    const { stdout, exitCode } = await runCLI(["create", "image", "--help"]);
    expect(exitCode).toBe(0);
    expect(stdout).toContain("--models-dir");
  });
});

describe("parallax CLI — anima image generation (US-002)", () => {
  it("US-002-AC01: missing PYCOMFY_MODELS_DIR and --models-dir exits 1 with clear error", async () => {
    const { stderr, exitCode } = await runCLIWithEnv(
      ["create", "image", "--model", "anima", "--prompt", "1girl, anime style"],
      { PYCOMFY_MODELS_DIR: undefined, PARALLAX_REPO_ROOT: undefined },
    );
    expect(exitCode).toBe(1);
    expect(stderr).toContain("Error: --models-dir or PYCOMFY_MODELS_DIR is required");
  });

  it("US-002-AC01: --models-dir flag takes precedence; missing PARALLAX_REPO_ROOT exits 1", async () => {
    const { stderr, exitCode } = await runCLIWithEnv(
      ["create", "image", "--model", "anima", "--prompt", "1girl, anime style", "--models-dir", "/tmp"],
      { PARALLAX_REPO_ROOT: undefined },
    );
    expect(exitCode).toBe(1);
    expect(stderr).toContain("Error: PARALLAX_REPO_ROOT is required");
  });

  it("US-002-AC02: --negative-prompt, --width, --height, --steps, --cfg, --seed are forwarded (reaches PARALLAX_REPO_ROOT check)", async () => {
    const { stderr, exitCode } = await runCLIWithEnv(
      [
        "create", "image", "--model", "anima", "--prompt", "1girl, anime style",
        "--models-dir", "/tmp", "--negative-prompt", "lowres, bad anatomy",
        "--width", "512", "--height", "512", "--steps", "10", "--cfg", "4", "--seed", "42",
        "--output", "out.png",
      ],
      { PARALLAX_REPO_ROOT: undefined },
    );
    // Flags are accepted; the only error should be the missing PARALLAX_REPO_ROOT
    expect(exitCode).toBe(1);
    expect(stderr).toContain("Error: PARALLAX_REPO_ROOT is required");
  });

  it("US-002-AC02: --negative-prompt IS forwarded for anima (unlike z_image)", async () => {
    const { stderr, exitCode } = await runCLIWithEnv(
      [
        "create", "image", "--model", "anima", "--prompt", "1girl",
        "--models-dir", "/tmp", "--negative-prompt", "bad quality",
      ],
      { PARALLAX_REPO_ROOT: undefined },
    );
    // Flag was accepted — no unknown-option error; only PARALLAX_REPO_ROOT error
    expect(exitCode).toBe(1);
    expect(stderr).not.toContain("unknown option");
    expect(stderr).toContain("Error: PARALLAX_REPO_ROOT is required");
  });

  it("US-002-AC03: CLI exits with non-zero when subprocess fails (bad PARALLAX_REPO_ROOT path)", async () => {
    const { exitCode } = await runCLIWithEnv(
      ["create", "image", "--model", "anima", "--prompt", "1girl", "--models-dir", "/tmp"],
      { PARALLAX_REPO_ROOT: "/nonexistent-parallax-root-12345" },
    );
    expect(exitCode).not.toBe(0);
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

// Helper: create a temporary PARALLAX_REPO_ROOT with a fake sdxl t2i.py script.
async function makeFakeSdxlRoot(scriptBody: string): Promise<string> {
  const tmpRoot = await mkdtemp(join(tmpdir(), "sdxl_test_"));
  const scriptDir = join(tmpRoot, "examples", "image", "generation", "sdxl");
  await mkdir(scriptDir, { recursive: true });
  await writeFile(join(scriptDir, "t2i.py"), scriptBody);
  return tmpRoot;
}

// Helper: create a temporary PARALLAX_REPO_ROOT with a fake z_image turbo.py script.
async function makeFakeZImageRoot(scriptBody: string): Promise<string> {
  const tmpRoot = await mkdtemp(join(tmpdir(), "z_image_test_"));
  const scriptDir = join(tmpRoot, "examples", "image", "generation", "z_image");
  await mkdir(scriptDir, { recursive: true });
  await writeFile(join(scriptDir, "turbo.py"), scriptBody);
  return tmpRoot;
}

describe("parallax CLI — z_image image generation (US-003-it39)", () => {
  // AC01: command spawns subprocess — verified by progressing past validation to
  // PARALLAX_REPO_ROOT check, which only happens after spawnPipeline is about to run.
  it("US-003-AC01: missing PYCOMFY_MODELS_DIR and --models-dir exits 1 with clear error", async () => {
    const { stderr, exitCode } = await runCLIWithEnv(
      ["create", "image", "--model", "z_image", "--prompt", "test"],
      { PYCOMFY_MODELS_DIR: undefined, PARALLAX_REPO_ROOT: undefined },
    );
    expect(exitCode).toBe(1);
    expect(stderr).toContain("Error: --models-dir or PYCOMFY_MODELS_DIR is required");
  });

  it("US-003-AC01: with --models-dir set, CLI reaches subprocess spawn (PARALLAX_REPO_ROOT check)", async () => {
    const { stderr, exitCode } = await runCLIWithEnv(
      ["create", "image", "--model", "z_image", "--prompt", "test", "--models-dir", "/tmp"],
      { PARALLAX_REPO_ROOT: undefined },
    );
    expect(exitCode).toBe(1);
    expect(stderr).toContain("Error: PARALLAX_REPO_ROOT is required");
  });

  it("US-003-AC01: subprocess is spawned and its exit code propagated (exit 0 from fake script)", async () => {
    const tmpRoot = await makeFakeZImageRoot("import sys; sys.exit(0)\n");
    try {
      const { exitCode } = await runCLIWithEnv(
        ["create", "image", "--model", "z_image", "--prompt", "test", "--models-dir", "/tmp"],
        { PARALLAX_REPO_ROOT: tmpRoot },
      );
      expect(exitCode).toBe(0);
    } finally {
      await rm(tmpRoot, { recursive: true, force: true });
    }
  });

  // AC02: --width, --height, --steps, --seed, --output are forwarded verbatim.
  it("US-003-AC02: --width, --height, --steps, --seed, --output are forwarded to the subprocess", async () => {
    // Fake script prints its argv so we can assert the flags were forwarded.
    const tmpRoot = await makeFakeZImageRoot(
      'import sys; print(" ".join(sys.argv[1:])); sys.exit(0)\n',
    );
    try {
      const { stdout, exitCode } = await runCLIWithEnv(
        [
          "create", "image", "--model", "z_image", "--prompt", "hello world",
          "--models-dir", "/tmp",
          "--width", "512", "--height", "768", "--steps", "4", "--seed", "99",
          "--output", "my_out.png",
        ],
        { PARALLAX_REPO_ROOT: tmpRoot },
      );
      expect(exitCode).toBe(0);
      expect(stdout).toContain("--width");
      expect(stdout).toContain("512");
      expect(stdout).toContain("--height");
      expect(stdout).toContain("768");
      expect(stdout).toContain("--steps");
      expect(stdout).toContain("4");
      expect(stdout).toContain("--seed");
      expect(stdout).toContain("99");
      expect(stdout).toContain("--output");
      expect(stdout).toContain("my_out.png");
    } finally {
      await rm(tmpRoot, { recursive: true, force: true });
    }
  });

  // AC03: --negative-prompt and --cfg are NOT forwarded to the z_image subprocess.
  it("US-003-AC03: --negative-prompt is NOT forwarded (subprocess exits 2 if it receives it, 0 otherwise)", async () => {
    const tmpRoot = await makeFakeZImageRoot(
      'import sys\nif "--negative-prompt" in sys.argv[1:]:\n    sys.exit(2)\nsys.exit(0)\n',
    );
    try {
      const { exitCode } = await runCLIWithEnv(
        [
          "create", "image", "--model", "z_image", "--prompt", "test",
          "--models-dir", "/tmp", "--negative-prompt", "bad quality",
        ],
        { PARALLAX_REPO_ROOT: tmpRoot },
      );
      // Would be 2 if --negative-prompt were forwarded; 0 confirms it was dropped.
      expect(exitCode).toBe(0);
    } finally {
      await rm(tmpRoot, { recursive: true, force: true });
    }
  });

  it("US-003-AC03: --cfg is NOT forwarded (subprocess exits 2 if it receives it, 0 otherwise)", async () => {
    const tmpRoot = await makeFakeZImageRoot(
      'import sys\nif "--cfg" in sys.argv[1:]:\n    sys.exit(2)\nsys.exit(0)\n',
    );
    try {
      const { exitCode } = await runCLIWithEnv(
        [
          "create", "image", "--model", "z_image", "--prompt", "test",
          "--models-dir", "/tmp", "--cfg", "7.5",
        ],
        { PARALLAX_REPO_ROOT: tmpRoot },
      );
      // Would be 2 if --cfg were forwarded; 0 confirms it was dropped.
      expect(exitCode).toBe(0);
    } finally {
      await rm(tmpRoot, { recursive: true, force: true });
    }
  });

  it("US-003-AC03: --negative-prompt and --cfg together are both NOT forwarded", async () => {
    const tmpRoot = await makeFakeZImageRoot(
      'import sys\nargs = sys.argv[1:]\nif "--negative-prompt" in args or "--cfg" in args:\n    sys.exit(2)\nsys.exit(0)\n',
    );
    try {
      const { exitCode } = await runCLIWithEnv(
        [
          "create", "image", "--model", "z_image", "--prompt", "test",
          "--models-dir", "/tmp", "--negative-prompt", "bad", "--cfg", "7",
        ],
        { PARALLAX_REPO_ROOT: tmpRoot },
      );
      expect(exitCode).toBe(0);
    } finally {
      await rm(tmpRoot, { recursive: true, force: true });
    }
  });

  // AC04: CLI exits with the subprocess exit code.
  it("US-003-AC04: CLI exits with non-zero when subprocess fails (bad PARALLAX_REPO_ROOT path)", async () => {
    const { exitCode } = await runCLIWithEnv(
      ["create", "image", "--model", "z_image", "--prompt", "test", "--models-dir", "/tmp"],
      { PARALLAX_REPO_ROOT: "/nonexistent-parallax-root-12345" },
    );
    expect(exitCode).not.toBe(0);
  });

  it("US-003-AC04: CLI propagates subprocess exit code 3 verbatim", async () => {
    const tmpRoot = await makeFakeZImageRoot("import sys; sys.exit(3)\n");
    try {
      const { exitCode } = await runCLIWithEnv(
        ["create", "image", "--model", "z_image", "--prompt", "test", "--models-dir", "/tmp"],
        { PARALLAX_REPO_ROOT: tmpRoot },
      );
      expect(exitCode).toBe(3);
    } finally {
      await rm(tmpRoot, { recursive: true, force: true });
    }
  });
});

describe("parallax CLI — models-dir resolution (US-004)", () => {
  // AC01: CLI reads PYCOMFY_MODELS_DIR from the environment when --models-dir is not supplied.
  it("US-004-AC01: PYCOMFY_MODELS_DIR env var is used when --models-dir is not given", async () => {
    const tmpRoot = await makeFakeSdxlRoot(
      'import sys; print(" ".join(sys.argv[1:])); sys.exit(0)\n',
    );
    try {
      const { stdout, exitCode } = await runCLIWithEnv(
        ["create", "image", "--model", "sdxl", "--prompt", "a cat"],
        { PARALLAX_REPO_ROOT: tmpRoot, PYCOMFY_MODELS_DIR: "/env/models" },
      );
      expect(exitCode).toBe(0);
      expect(stdout).toContain("--models-dir");
      expect(stdout).toContain("/env/models");
    } finally {
      await rm(tmpRoot, { recursive: true, force: true });
    }
  });

  // AC02: --models-dir flag takes precedence over PYCOMFY_MODELS_DIR.
  it("US-004-AC02: --models-dir flag overrides PYCOMFY_MODELS_DIR env var", async () => {
    const tmpRoot = await makeFakeSdxlRoot(
      'import sys; print(" ".join(sys.argv[1:])); sys.exit(0)\n',
    );
    try {
      const { stdout, exitCode } = await runCLIWithEnv(
        ["create", "image", "--model", "sdxl", "--prompt", "a cat", "--models-dir", "/flag/models"],
        { PARALLAX_REPO_ROOT: tmpRoot, PYCOMFY_MODELS_DIR: "/env/models" },
      );
      expect(exitCode).toBe(0);
      expect(stdout).toContain("--models-dir");
      expect(stdout).toContain("/flag/models");
      expect(stdout).not.toContain("/env/models");
    } finally {
      await rm(tmpRoot, { recursive: true, force: true });
    }
  });

  // AC03: If neither --models-dir nor PYCOMFY_MODELS_DIR is set, CLI prints a clear error.
  it("US-004-AC03: missing both --models-dir and PYCOMFY_MODELS_DIR exits 1 with clear error", async () => {
    const { stderr, exitCode } = await runCLIWithEnv(
      ["create", "image", "--model", "sdxl", "--prompt", "a cat"],
      { PYCOMFY_MODELS_DIR: undefined, PARALLAX_REPO_ROOT: undefined },
    );
    expect(exitCode).toBe(1);
    expect(stderr).toContain("Error: --models-dir or PYCOMFY_MODELS_DIR is required");
  });

  // AC04: The resolved path is forwarded to the subprocess as --models-dir <path>.
  it("US-004-AC04: resolved models-dir from env is passed as --models-dir to subprocess", async () => {
    const tmpRoot = await makeFakeSdxlRoot(
      'import sys; print(" ".join(sys.argv[1:])); sys.exit(0)\n',
    );
    try {
      const { stdout, exitCode } = await runCLIWithEnv(
        ["create", "image", "--model", "sdxl", "--prompt", "a cat"],
        { PARALLAX_REPO_ROOT: tmpRoot, PYCOMFY_MODELS_DIR: "/resolved/models" },
      );
      expect(exitCode).toBe(0);
      // The subprocess argv must contain "--models-dir /resolved/models"
      const idx = stdout.split(" ").indexOf("--models-dir");
      expect(idx).toBeGreaterThanOrEqual(0);
      expect(stdout.split(" ")[idx + 1]?.trim()).toBe("/resolved/models");
    } finally {
      await rm(tmpRoot, { recursive: true, force: true });
    }
  });

  it("US-004-AC04: resolved models-dir from flag is passed as --models-dir to subprocess", async () => {
    const tmpRoot = await makeFakeSdxlRoot(
      'import sys; print(" ".join(sys.argv[1:])); sys.exit(0)\n',
    );
    try {
      const { stdout, exitCode } = await runCLIWithEnv(
        ["create", "image", "--model", "sdxl", "--prompt", "a cat", "--models-dir", "/flag/models"],
        { PARALLAX_REPO_ROOT: tmpRoot, PYCOMFY_MODELS_DIR: undefined },
      );
      expect(exitCode).toBe(0);
      const parts = stdout.split(" ");
      const idx = parts.indexOf("--models-dir");
      expect(idx).toBeGreaterThanOrEqual(0);
      expect(parts[idx + 1]?.trim()).toBe("/flag/models");
    } finally {
      await rm(tmpRoot, { recursive: true, force: true });
    }
  });
});

// Helper: create a temporary PARALLAX_REPO_ROOT with a fake ltx23 t2v.py script.
async function makeFakeLtx23Root(scriptBody: string): Promise<string> {
  const tmpRoot = await mkdtemp(join(tmpdir(), "ltx23_test_"));
  const scriptDir = join(tmpRoot, "examples", "video", "ltx", "ltx23");
  await mkdir(scriptDir, { recursive: true });
  await writeFile(join(scriptDir, "t2v.py"), scriptBody);
  return tmpRoot;
}

// Helper: create a temporary PARALLAX_REPO_ROOT with a fake ltx2 t2v.py script.
async function makeFakeLtx2Root(scriptBody: string): Promise<string> {
  const tmpRoot = await mkdtemp(join(tmpdir(), "ltx2_test_"));
  const scriptDir = join(tmpRoot, "examples", "video", "ltx", "ltx2");
  await mkdir(scriptDir, { recursive: true });
  await writeFile(join(scriptDir, "t2v.py"), scriptBody);
  return tmpRoot;
}

describe("parallax CLI — ltx2 video generation (US-001)", () => {
  // AC01: missing models-dir exits early with clear error
  it("US-001-AC01: missing PYCOMFY_MODELS_DIR and --models-dir exits 1 with clear error", async () => {
    const { stderr, exitCode } = await runCLIWithEnv(
      ["create", "video", "--model", "ltx2", "--prompt", "test"],
      { PYCOMFY_MODELS_DIR: undefined, PARALLAX_REPO_ROOT: undefined },
    );
    expect(exitCode).toBe(1);
    expect(stderr).toContain("Error: --models-dir or PYCOMFY_MODELS_DIR is required");
  });

  it("US-001-AC01: with --models-dir set, CLI reaches subprocess spawn (PARALLAX_REPO_ROOT check)", async () => {
    const { stderr, exitCode } = await runCLIWithEnv(
      ["create", "video", "--model", "ltx2", "--prompt", "test", "--models-dir", "/tmp"],
      { PARALLAX_REPO_ROOT: undefined },
    );
    expect(exitCode).toBe(1);
    expect(stderr).toContain("Error: PARALLAX_REPO_ROOT is required");
  });

  it("US-001-AC01: subprocess is spawned and its exit code propagated (exit 0 from fake script)", async () => {
    const tmpRoot = await makeFakeLtx2Root("import sys; sys.exit(0)\n");
    try {
      const { exitCode } = await runCLIWithEnv(
        ["create", "video", "--model", "ltx2", "--prompt", "test", "--models-dir", "/tmp"],
        { PARALLAX_REPO_ROOT: tmpRoot },
      );
      expect(exitCode).toBe(0);
    } finally {
      await rm(tmpRoot, { recursive: true, force: true });
    }
  });

  // AC02: --prompt, --width, --height, --length, --seed, --output are forwarded verbatim
  it("US-001-AC02: --prompt, --width, --height, --length, --seed, --output are forwarded to subprocess", async () => {
    const tmpRoot = await makeFakeLtx2Root(
      'import sys; print(" ".join(sys.argv[1:])); sys.exit(0)\n',
    );
    try {
      const { stdout, exitCode } = await runCLIWithEnv(
        [
          "create", "video", "--model", "ltx2", "--prompt", "a cat video",
          "--models-dir", "/tmp",
          "--width", "1280", "--height", "720", "--length", "97",
          "--seed", "42", "--output", "my_video.mp4",
        ],
        { PARALLAX_REPO_ROOT: tmpRoot },
      );
      expect(exitCode).toBe(0);
      expect(stdout).toContain("--prompt");
      expect(stdout).toContain("a cat video");
      expect(stdout).toContain("--width");
      expect(stdout).toContain("1280");
      expect(stdout).toContain("--height");
      expect(stdout).toContain("720");
      expect(stdout).toContain("--length");
      expect(stdout).toContain("97");
      expect(stdout).toContain("--seed");
      expect(stdout).toContain("42");
      expect(stdout).toContain("--output");
      expect(stdout).toContain("my_video.mp4");
    } finally {
      await rm(tmpRoot, { recursive: true, force: true });
    }
  });

  // AC03: --cfg is forwarded as --cfg-pass1, never as bare --cfg
  it("US-001-AC03: --cfg is forwarded as --cfg-pass1 to ltx2 subprocess", async () => {
    const tmpRoot = await makeFakeLtx2Root(
      'import sys; print(" ".join(sys.argv[1:])); sys.exit(0)\n',
    );
    try {
      const { stdout, exitCode } = await runCLIWithEnv(
        ["create", "video", "--model", "ltx2", "--prompt", "test", "--models-dir", "/tmp", "--cfg", "4.5"],
        { PARALLAX_REPO_ROOT: tmpRoot },
      );
      expect(exitCode).toBe(0);
      expect(stdout).toContain("--cfg-pass1");
      expect(stdout).toContain("4.5");
      // "--cfg " (with trailing space) is the bare flag — it must not appear
      expect(stdout).not.toContain("--cfg ");
    } finally {
      await rm(tmpRoot, { recursive: true, force: true });
    }
  });

  it("US-001-AC03: bare --cfg is NOT present in subprocess argv (exact element check)", async () => {
    const tmpRoot = await makeFakeLtx2Root(
      'import sys\nif "--cfg" in sys.argv[1:]:\n    sys.exit(2)\nsys.exit(0)\n',
    );
    try {
      const { exitCode } = await runCLIWithEnv(
        ["create", "video", "--model", "ltx2", "--prompt", "test", "--models-dir", "/tmp", "--cfg", "4.5"],
        { PARALLAX_REPO_ROOT: tmpRoot },
      );
      // Would be 2 if bare --cfg were forwarded; 0 confirms it was converted to --cfg-pass1.
      expect(exitCode).toBe(0);
    } finally {
      await rm(tmpRoot, { recursive: true, force: true });
    }
  });

  // AC04: --steps is forwarded as --steps
  it("US-001-AC04: --steps is forwarded as --steps to ltx2 subprocess", async () => {
    const tmpRoot = await makeFakeLtx2Root(
      'import sys; print(" ".join(sys.argv[1:])); sys.exit(0)\n',
    );
    try {
      const { stdout, exitCode } = await runCLIWithEnv(
        ["create", "video", "--model", "ltx2", "--prompt", "test", "--models-dir", "/tmp", "--steps", "15"],
        { PARALLAX_REPO_ROOT: tmpRoot },
      );
      expect(exitCode).toBe(0);
      expect(stdout).toContain("--steps");
      expect(stdout).toContain("15");
    } finally {
      await rm(tmpRoot, { recursive: true, force: true });
    }
  });

  // AC05: --models-dir is forwarded to the subprocess
  it("US-001-AC05: --models-dir is forwarded to the ltx2 subprocess", async () => {
    const tmpRoot = await makeFakeLtx2Root(
      'import sys; print(" ".join(sys.argv[1:])); sys.exit(0)\n',
    );
    try {
      const { stdout, exitCode } = await runCLIWithEnv(
        ["create", "video", "--model", "ltx2", "--prompt", "test", "--models-dir", "/my/models"],
        { PARALLAX_REPO_ROOT: tmpRoot },
      );
      expect(exitCode).toBe(0);
      expect(stdout).toContain("--models-dir");
      expect(stdout).toContain("/my/models");
    } finally {
      await rm(tmpRoot, { recursive: true, force: true });
    }
  });

  it("US-001-AC05: --models-dir option is listed in create video --help", async () => {
    const { stdout, exitCode } = await runCLI(["create", "video", "--help"]);
    expect(exitCode).toBe(0);
    expect(stdout).toContain("--models-dir");
  });

  // AC06: CLI exits with the subprocess exit code
  it("US-001-AC06: CLI exits with non-zero when subprocess fails (bad PARALLAX_REPO_ROOT path)", async () => {
    const { exitCode } = await runCLIWithEnv(
      ["create", "video", "--model", "ltx2", "--prompt", "test", "--models-dir", "/tmp"],
      { PARALLAX_REPO_ROOT: "/nonexistent-parallax-root-12345" },
    );
    expect(exitCode).not.toBe(0);
  });

  it("US-001-AC06: CLI propagates subprocess exit code 3 verbatim", async () => {
    const tmpRoot = await makeFakeLtx2Root("import sys; sys.exit(3)\n");
    try {
      const { exitCode } = await runCLIWithEnv(
        ["create", "video", "--model", "ltx2", "--prompt", "test", "--models-dir", "/tmp"],
        { PARALLAX_REPO_ROOT: tmpRoot },
      );
      expect(exitCode).toBe(3);
    } finally {
      await rm(tmpRoot, { recursive: true, force: true });
    }
  });
});

describe("parallax CLI — ltx23 video generation (US-002-it39)", () => {
  // AC01: running the command spawns uv run python examples/video/ltx/ltx23/t2v.py
  it("US-002-AC01: missing PYCOMFY_MODELS_DIR and --models-dir exits 1 with clear error", async () => {
    const { stderr, exitCode } = await runCLIWithEnv(
      ["create", "video", "--model", "ltx23", "--prompt", "test"],
      { PYCOMFY_MODELS_DIR: undefined, PARALLAX_REPO_ROOT: undefined },
    );
    expect(exitCode).toBe(1);
    expect(stderr).toContain("Error: --models-dir or PYCOMFY_MODELS_DIR is required");
  });

  it("US-002-AC01: with --models-dir set, CLI reaches subprocess spawn (PARALLAX_REPO_ROOT check)", async () => {
    const { stderr, exitCode } = await runCLIWithEnv(
      ["create", "video", "--model", "ltx23", "--prompt", "test", "--models-dir", "/tmp"],
      { PARALLAX_REPO_ROOT: undefined },
    );
    expect(exitCode).toBe(1);
    expect(stderr).toContain("Error: PARALLAX_REPO_ROOT is required");
  });

  it("US-002-AC01: subprocess is spawned and its exit code propagated (exit 0 from fake script)", async () => {
    const tmpRoot = await makeFakeLtx23Root("import sys; sys.exit(0)\n");
    try {
      const { exitCode } = await runCLIWithEnv(
        ["create", "video", "--model", "ltx23", "--prompt", "test", "--models-dir", "/tmp"],
        { PARALLAX_REPO_ROOT: tmpRoot },
      );
      expect(exitCode).toBe(0);
    } finally {
      await rm(tmpRoot, { recursive: true, force: true });
    }
  });

  // AC02: --prompt, --width, --height, --length, --cfg, --seed, --output are forwarded
  it("US-002-AC02: --prompt, --width, --height, --length, --cfg, --seed, --output are forwarded to subprocess", async () => {
    const tmpRoot = await makeFakeLtx23Root(
      'import sys; print(" ".join(sys.argv[1:])); sys.exit(0)\n',
    );
    try {
      const { stdout, exitCode } = await runCLIWithEnv(
        [
          "create", "video", "--model", "ltx23", "--prompt", "a jazz video",
          "--models-dir", "/tmp",
          "--width", "768", "--height", "512", "--length", "97",
          "--cfg", "1.0", "--seed", "7", "--output", "out.mp4",
        ],
        { PARALLAX_REPO_ROOT: tmpRoot },
      );
      expect(exitCode).toBe(0);
      expect(stdout).toContain("--prompt");
      expect(stdout).toContain("a jazz video");
      expect(stdout).toContain("--width");
      expect(stdout).toContain("768");
      expect(stdout).toContain("--height");
      expect(stdout).toContain("512");
      expect(stdout).toContain("--length");
      expect(stdout).toContain("97");
      expect(stdout).toContain("--cfg");
      expect(stdout).toContain("1.0");
      expect(stdout).toContain("--seed");
      expect(stdout).toContain("7");
      expect(stdout).toContain("--output");
      expect(stdout).toContain("out.mp4");
    } finally {
      await rm(tmpRoot, { recursive: true, force: true });
    }
  });

  // AC03: --steps is NOT forwarded (ltx23 t2v is distilled)
  it("US-002-AC03: --steps is NOT forwarded to ltx23 subprocess", async () => {
    const tmpRoot = await makeFakeLtx23Root(
      'import sys\nif "--steps" in sys.argv[1:]:\n    sys.exit(2)\nsys.exit(0)\n',
    );
    try {
      const { exitCode } = await runCLIWithEnv(
        ["create", "video", "--model", "ltx23", "--prompt", "test", "--models-dir", "/tmp", "--steps", "20"],
        { PARALLAX_REPO_ROOT: tmpRoot },
      );
      // Would be 2 if --steps were forwarded; 0 confirms it was dropped.
      expect(exitCode).toBe(0);
    } finally {
      await rm(tmpRoot, { recursive: true, force: true });
    }
  });

  it("US-002-AC03: --steps is absent even when not explicitly passed (default not forwarded)", async () => {
    const tmpRoot = await makeFakeLtx23Root(
      'import sys\nif "--steps" in sys.argv[1:]:\n    sys.exit(2)\nsys.exit(0)\n',
    );
    try {
      const { exitCode } = await runCLIWithEnv(
        ["create", "video", "--model", "ltx23", "--prompt", "test", "--models-dir", "/tmp"],
        { PARALLAX_REPO_ROOT: tmpRoot },
      );
      expect(exitCode).toBe(0);
    } finally {
      await rm(tmpRoot, { recursive: true, force: true });
    }
  });

  // AC04: --models-dir is forwarded
  it("US-002-AC04: --models-dir is forwarded to the ltx23 subprocess", async () => {
    const tmpRoot = await makeFakeLtx23Root(
      'import sys; print(" ".join(sys.argv[1:])); sys.exit(0)\n',
    );
    try {
      const { stdout, exitCode } = await runCLIWithEnv(
        ["create", "video", "--model", "ltx23", "--prompt", "test", "--models-dir", "/my/models"],
        { PARALLAX_REPO_ROOT: tmpRoot },
      );
      expect(exitCode).toBe(0);
      expect(stdout).toContain("--models-dir");
      expect(stdout).toContain("/my/models");
    } finally {
      await rm(tmpRoot, { recursive: true, force: true });
    }
  });

  it("US-002-AC04: --models-dir option is listed in create video --help", async () => {
    const { stdout, exitCode } = await runCLI(["create", "video", "--help"]);
    expect(exitCode).toBe(0);
    expect(stdout).toContain("--models-dir");
  });

  // AC05: CLI exits with the subprocess exit code
  it("US-002-AC05: CLI exits with non-zero when subprocess fails (bad PARALLAX_REPO_ROOT path)", async () => {
    const { exitCode } = await runCLIWithEnv(
      ["create", "video", "--model", "ltx23", "--prompt", "test", "--models-dir", "/tmp"],
      { PARALLAX_REPO_ROOT: "/nonexistent-parallax-root-12345" },
    );
    expect(exitCode).not.toBe(0);
  });

  it("US-002-AC05: CLI propagates subprocess exit code 3 verbatim", async () => {
    const tmpRoot = await makeFakeLtx23Root("import sys; sys.exit(3)\n");
    try {
      const { exitCode } = await runCLIWithEnv(
        ["create", "video", "--model", "ltx23", "--prompt", "test", "--models-dir", "/tmp"],
        { PARALLAX_REPO_ROOT: tmpRoot },
      );
      expect(exitCode).toBe(3);
    } finally {
      await rm(tmpRoot, { recursive: true, force: true });
    }
  });
});

// Helper: create a temporary PARALLAX_REPO_ROOT with a fake wan21 t2v.py script.
async function makeFakeWan21Root(scriptBody: string): Promise<string> {
  const tmpRoot = await mkdtemp(join(tmpdir(), "wan21_test_"));
  const scriptDir = join(tmpRoot, "examples", "video", "wan", "wan21");
  await mkdir(scriptDir, { recursive: true });
  await writeFile(join(scriptDir, "t2v.py"), scriptBody);
  return tmpRoot;
}

describe("parallax CLI — wan21 video generation (US-003)", () => {
  // AC01: running the command spawns uv run python examples/video/wan/wan21/t2v.py
  it("US-003-AC01: missing PYCOMFY_MODELS_DIR and --models-dir exits 1 with clear error", async () => {
    const { stderr, exitCode } = await runCLIWithEnv(
      ["create", "video", "--model", "wan21", "--prompt", "test"],
      { PYCOMFY_MODELS_DIR: undefined, PARALLAX_REPO_ROOT: undefined },
    );
    expect(exitCode).toBe(1);
    expect(stderr).toContain("Error: --models-dir or PYCOMFY_MODELS_DIR is required");
  });

  it("US-003-AC01: with --models-dir set, CLI reaches subprocess spawn (PARALLAX_REPO_ROOT check)", async () => {
    const { stderr, exitCode } = await runCLIWithEnv(
      ["create", "video", "--model", "wan21", "--prompt", "test", "--models-dir", "/tmp"],
      { PARALLAX_REPO_ROOT: undefined },
    );
    expect(exitCode).toBe(1);
    expect(stderr).toContain("Error: PARALLAX_REPO_ROOT is required");
  });

  it("US-003-AC01: subprocess is spawned and its exit code propagated (exit 0 from fake script)", async () => {
    const tmpRoot = await makeFakeWan21Root("import sys; sys.exit(0)\n");
    try {
      const { exitCode } = await runCLIWithEnv(
        ["create", "video", "--model", "wan21", "--prompt", "test", "--models-dir", "/tmp"],
        { PARALLAX_REPO_ROOT: tmpRoot },
      );
      expect(exitCode).toBe(0);
    } finally {
      await rm(tmpRoot, { recursive: true, force: true });
    }
  });

  // AC02: --prompt, --width, --height, --length, --steps, --cfg, --seed are forwarded
  it("US-003-AC02: --prompt, --width, --height, --length, --steps, --cfg, --seed, --output are forwarded to subprocess", async () => {
    const tmpRoot = await makeFakeWan21Root(
      'import sys; print(" ".join(sys.argv[1:])); sys.exit(0)\n',
    );
    try {
      const { stdout, exitCode } = await runCLIWithEnv(
        [
          "create", "video", "--model", "wan21", "--prompt", "a winter fox",
          "--models-dir", "/tmp",
          "--width", "832", "--height", "480", "--length", "33",
          "--steps", "30", "--cfg", "6", "--seed", "7", "--output", "fox.mp4",
        ],
        { PARALLAX_REPO_ROOT: tmpRoot },
      );
      expect(exitCode).toBe(0);
      expect(stdout).toContain("--prompt");
      expect(stdout).toContain("a winter fox");
      expect(stdout).toContain("--width");
      expect(stdout).toContain("832");
      expect(stdout).toContain("--height");
      expect(stdout).toContain("480");
      expect(stdout).toContain("--length");
      expect(stdout).toContain("33");
      expect(stdout).toContain("--steps");
      expect(stdout).toContain("30");
      expect(stdout).toContain("--cfg");
      expect(stdout).toContain("6");
      expect(stdout).toContain("--seed");
      expect(stdout).toContain("7");
      expect(stdout).toContain("--output");
      expect(stdout).toContain("fox.mp4");
    } finally {
      await rm(tmpRoot, { recursive: true, force: true });
    }
  });

  // AC03: --models-dir is forwarded
  it("US-003-AC03: --models-dir is forwarded to the wan21 subprocess", async () => {
    const tmpRoot = await makeFakeWan21Root(
      'import sys; print(" ".join(sys.argv[1:])); sys.exit(0)\n',
    );
    try {
      const { stdout, exitCode } = await runCLIWithEnv(
        ["create", "video", "--model", "wan21", "--prompt", "test", "--models-dir", "/my/models"],
        { PARALLAX_REPO_ROOT: tmpRoot },
      );
      expect(exitCode).toBe(0);
      expect(stdout).toContain("--models-dir");
      expect(stdout).toContain("/my/models");
    } finally {
      await rm(tmpRoot, { recursive: true, force: true });
    }
  });

  it("US-003-AC03: --models-dir option is listed in create video --help", async () => {
    const { stdout, exitCode } = await runCLI(["create", "video", "--help"]);
    expect(exitCode).toBe(0);
    expect(stdout).toContain("--models-dir");
  });

  // AC04: CLI exits with the subprocess exit code
  it("US-003-AC04: CLI exits with non-zero when subprocess fails (bad PARALLAX_REPO_ROOT path)", async () => {
    const { exitCode } = await runCLIWithEnv(
      ["create", "video", "--model", "wan21", "--prompt", "test", "--models-dir", "/tmp"],
      { PARALLAX_REPO_ROOT: "/nonexistent-parallax-root-12345" },
    );
    expect(exitCode).not.toBe(0);
  });

  it("US-003-AC04: CLI propagates subprocess exit code 3 verbatim", async () => {
    const tmpRoot = await makeFakeWan21Root("import sys; sys.exit(3)\n");
    try {
      const { exitCode } = await runCLIWithEnv(
        ["create", "video", "--model", "wan21", "--prompt", "test", "--models-dir", "/tmp"],
        { PARALLAX_REPO_ROOT: tmpRoot },
      );
      expect(exitCode).toBe(3);
    } finally {
      await rm(tmpRoot, { recursive: true, force: true });
    }
  });
});

describe("parallax CLI — models-dir and repo-root resolution for create video (US-005)", () => {
  // AC01: --models-dir flag takes precedence over PYCOMFY_MODELS_DIR
  it("US-005-AC01: --models-dir flag overrides PYCOMFY_MODELS_DIR for create video", async () => {
    const tmpRoot = await makeFakeWan21Root(
      'import sys; print(" ".join(sys.argv[1:])); sys.exit(0)\n',
    );
    try {
      const { stdout, exitCode } = await runCLIWithEnv(
        ["create", "video", "--model", "wan21", "--prompt", "test", "--models-dir", "/flag/models"],
        { PARALLAX_REPO_ROOT: tmpRoot, PYCOMFY_MODELS_DIR: "/env/models" },
      );
      expect(exitCode).toBe(0);
      expect(stdout).toContain("--models-dir");
      expect(stdout).toContain("/flag/models");
      expect(stdout).not.toContain("/env/models");
    } finally {
      await rm(tmpRoot, { recursive: true, force: true });
    }
  });

  // AC02: If neither --models-dir nor PYCOMFY_MODELS_DIR is set, CLI prints error
  it("US-005-AC02: missing both --models-dir and PYCOMFY_MODELS_DIR exits 1 with clear error", async () => {
    const { stderr, exitCode } = await runCLIWithEnv(
      ["create", "video", "--model", "wan21", "--prompt", "test"],
      { PYCOMFY_MODELS_DIR: undefined, PARALLAX_REPO_ROOT: undefined },
    );
    expect(exitCode).toBe(1);
    expect(stderr).toContain("Error: --models-dir or PYCOMFY_MODELS_DIR is required");
  });

  // AC03: If PARALLAX_REPO_ROOT is not set, CLI prints error (models-dir already resolved)
  it("US-005-AC03: missing PARALLAX_REPO_ROOT exits 1 with clear error", async () => {
    const { stderr, exitCode } = await runCLIWithEnv(
      ["create", "video", "--model", "wan21", "--prompt", "test", "--models-dir", "/tmp"],
      { PARALLAX_REPO_ROOT: undefined },
    );
    expect(exitCode).toBe(1);
    expect(stderr).toContain("Error: PARALLAX_REPO_ROOT is required");
  });

  it("US-005-AC03: PYCOMFY_MODELS_DIR set but PARALLAX_REPO_ROOT missing still exits 1", async () => {
    const { stderr, exitCode } = await runCLIWithEnv(
      ["create", "video", "--model", "wan21", "--prompt", "test"],
      { PYCOMFY_MODELS_DIR: "/env/models", PARALLAX_REPO_ROOT: undefined },
    );
    expect(exitCode).toBe(1);
    expect(stderr).toContain("Error: PARALLAX_REPO_ROOT is required");
  });

  // AC04: The resolved models path is passed as --models-dir <path> to subprocess
  it("US-005-AC04: PYCOMFY_MODELS_DIR env var is passed as --models-dir to subprocess", async () => {
    const tmpRoot = await makeFakeWan21Root(
      'import sys; print(" ".join(sys.argv[1:])); sys.exit(0)\n',
    );
    try {
      const { stdout, exitCode } = await runCLIWithEnv(
        ["create", "video", "--model", "wan21", "--prompt", "test"],
        { PARALLAX_REPO_ROOT: tmpRoot, PYCOMFY_MODELS_DIR: "/env/video/models" },
      );
      expect(exitCode).toBe(0);
      const parts = stdout.split(" ");
      const idx = parts.indexOf("--models-dir");
      expect(idx).toBeGreaterThanOrEqual(0);
      expect(parts[idx + 1]?.trim()).toBe("/env/video/models");
    } finally {
      await rm(tmpRoot, { recursive: true, force: true });
    }
  });

  it("US-005-AC04: --models-dir flag value is passed as --models-dir to subprocess", async () => {
    const tmpRoot = await makeFakeWan21Root(
      'import sys; print(" ".join(sys.argv[1:])); sys.exit(0)\n',
    );
    try {
      const { stdout, exitCode } = await runCLIWithEnv(
        ["create", "video", "--model", "wan21", "--prompt", "test", "--models-dir", "/flag/video/models"],
        { PARALLAX_REPO_ROOT: tmpRoot, PYCOMFY_MODELS_DIR: undefined },
      );
      expect(exitCode).toBe(0);
      const parts = stdout.split(" ");
      const idx = parts.indexOf("--models-dir");
      expect(idx).toBeGreaterThanOrEqual(0);
      expect(parts[idx + 1]?.trim()).toBe("/flag/video/models");
    } finally {
      await rm(tmpRoot, { recursive: true, force: true });
    }
  });
});

// Helper: create a temporary PARALLAX_REPO_ROOT with a fake wan22 t2v.py script.
async function makeFakeWan22Root(scriptBody: string): Promise<string> {
  const tmpRoot = await mkdtemp(join(tmpdir(), "wan22_test_"));
  const scriptDir = join(tmpRoot, "examples", "video", "wan", "wan22");
  await mkdir(scriptDir, { recursive: true });
  await writeFile(join(scriptDir, "t2v.py"), scriptBody);
  return tmpRoot;
}

// Helper: create a temporary PARALLAX_REPO_ROOT with a fake wan22 i2v.py script.
async function makeFakeWan22I2vRoot(scriptBody: string): Promise<string> {
  const tmpRoot = await mkdtemp(join(tmpdir(), "wan22_i2v_test_"));
  const scriptDir = join(tmpRoot, "examples", "video", "wan", "wan22");
  await mkdir(scriptDir, { recursive: true });
  await writeFile(join(scriptDir, "i2v.py"), scriptBody);
  return tmpRoot;
}

describe("parallax CLI — wan22 video generation (US-004)", () => {
  // AC01: running the command spawns uv run python examples/video/wan/wan22/t2v.py
  it("US-004-AC01: missing PYCOMFY_MODELS_DIR and --models-dir exits 1 with clear error", async () => {
    const { stderr, exitCode } = await runCLIWithEnv(
      ["create", "video", "--model", "wan22", "--prompt", "test"],
      { PYCOMFY_MODELS_DIR: undefined, PARALLAX_REPO_ROOT: undefined },
    );
    expect(exitCode).toBe(1);
    expect(stderr).toContain("Error: --models-dir or PYCOMFY_MODELS_DIR is required");
  });

  it("US-004-AC01: with --models-dir set, CLI reaches subprocess spawn (PARALLAX_REPO_ROOT check)", async () => {
    const { stderr, exitCode } = await runCLIWithEnv(
      ["create", "video", "--model", "wan22", "--prompt", "test", "--models-dir", "/tmp"],
      { PARALLAX_REPO_ROOT: undefined },
    );
    expect(exitCode).toBe(1);
    expect(stderr).toContain("Error: PARALLAX_REPO_ROOT is required");
  });

  it("US-004-AC01: subprocess is spawned and its exit code propagated (exit 0 from fake script)", async () => {
    const tmpRoot = await makeFakeWan22Root("import sys; sys.exit(0)\n");
    try {
      const { exitCode } = await runCLIWithEnv(
        ["create", "video", "--model", "wan22", "--prompt", "test", "--models-dir", "/tmp"],
        { PARALLAX_REPO_ROOT: tmpRoot },
      );
      expect(exitCode).toBe(0);
    } finally {
      await rm(tmpRoot, { recursive: true, force: true });
    }
  });

  // AC02: --prompt, --width, --height, --length, --steps, --cfg, --seed are forwarded
  it("US-004-AC02: --prompt, --width, --height, --length, --steps, --cfg, --seed, --output are forwarded to subprocess", async () => {
    const tmpRoot = await makeFakeWan22Root(
      'import sys; print(" ".join(sys.argv[1:])); sys.exit(0)\n',
    );
    try {
      const { stdout, exitCode } = await runCLIWithEnv(
        [
          "create", "video", "--model", "wan22", "--prompt", "a mountain river",
          "--models-dir", "/tmp",
          "--width", "832", "--height", "480", "--length", "81",
          "--steps", "4", "--cfg", "1", "--seed", "99", "--output", "river.mp4",
        ],
        { PARALLAX_REPO_ROOT: tmpRoot },
      );
      expect(exitCode).toBe(0);
      expect(stdout).toContain("--prompt");
      expect(stdout).toContain("a mountain river");
      expect(stdout).toContain("--width");
      expect(stdout).toContain("832");
      expect(stdout).toContain("--height");
      expect(stdout).toContain("480");
      expect(stdout).toContain("--length");
      expect(stdout).toContain("81");
      expect(stdout).toContain("--steps");
      expect(stdout).toContain("4");
      expect(stdout).toContain("--cfg");
      expect(stdout).toContain("1");
      expect(stdout).toContain("--seed");
      expect(stdout).toContain("99");
      expect(stdout).toContain("--output");
      expect(stdout).toContain("river.mp4");
    } finally {
      await rm(tmpRoot, { recursive: true, force: true });
    }
  });

  // AC03: --models-dir is forwarded
  it("US-004-AC03: --models-dir is forwarded to the wan22 subprocess", async () => {
    const tmpRoot = await makeFakeWan22Root(
      'import sys; print(" ".join(sys.argv[1:])); sys.exit(0)\n',
    );
    try {
      const { stdout, exitCode } = await runCLIWithEnv(
        ["create", "video", "--model", "wan22", "--prompt", "test", "--models-dir", "/my/models"],
        { PARALLAX_REPO_ROOT: tmpRoot },
      );
      expect(exitCode).toBe(0);
      expect(stdout).toContain("--models-dir");
      expect(stdout).toContain("/my/models");
    } finally {
      await rm(tmpRoot, { recursive: true, force: true });
    }
  });

  it("US-004-AC03: --models-dir option is listed in create video --help", async () => {
    const { stdout, exitCode } = await runCLI(["create", "video", "--help"]);
    expect(exitCode).toBe(0);
    expect(stdout).toContain("--models-dir");
  });

  // AC04: CLI exits with the subprocess exit code
  it("US-004-AC04: CLI exits with non-zero when subprocess fails (bad PARALLAX_REPO_ROOT path)", async () => {
    const { exitCode } = await runCLIWithEnv(
      ["create", "video", "--model", "wan22", "--prompt", "test", "--models-dir", "/tmp"],
      { PARALLAX_REPO_ROOT: "/nonexistent-parallax-root-12345" },
    );
    expect(exitCode).not.toBe(0);
  });

  it("US-004-AC04: CLI propagates subprocess exit code 3 verbatim", async () => {
    const tmpRoot = await makeFakeWan22Root("import sys; sys.exit(3)\n");
    try {
      const { exitCode } = await runCLIWithEnv(
        ["create", "video", "--model", "wan22", "--prompt", "test", "--models-dir", "/tmp"],
        { PARALLAX_REPO_ROOT: tmpRoot },
      );
      expect(exitCode).toBe(3);
    } finally {
      await rm(tmpRoot, { recursive: true, force: true });
    }
  });
});

// Helper: create a temporary PARALLAX_REPO_ROOT with a fake ltx2 i2v.py script.
async function makeFakeLtx2I2vRoot(scriptBody: string): Promise<string> {
  const tmpRoot = await mkdtemp(join(tmpdir(), "ltx2_i2v_test_"));
  const scriptDir = join(tmpRoot, "examples", "video", "ltx", "ltx2");
  await mkdir(scriptDir, { recursive: true });
  await writeFile(join(scriptDir, "i2v.py"), scriptBody);
  return tmpRoot;
}

describe("parallax CLI — ltx2 i2v video generation (US-001-it39)", () => {
  // AC01: missing models-dir exits early with clear error
  it("US-001-AC01: missing PYCOMFY_MODELS_DIR with --input exits 1 with clear error", async () => {
    const { stderr, exitCode } = await runCLIWithEnv(
      ["create", "video", "--model", "ltx2", "--prompt", "test", "--input", "/etc/hostname"],
      { PYCOMFY_MODELS_DIR: undefined, PARALLAX_REPO_ROOT: undefined },
    );
    expect(exitCode).toBe(1);
    expect(stderr).toContain("Error: --models-dir or PYCOMFY_MODELS_DIR is required");
  });

  it("US-001-AC01: with --input and --models-dir set, CLI reaches subprocess spawn (PARALLAX_REPO_ROOT check)", async () => {
    const { stderr, exitCode } = await runCLIWithEnv(
      ["create", "video", "--model", "ltx2", "--prompt", "test", "--input", "/etc/hostname", "--models-dir", "/tmp"],
      { PARALLAX_REPO_ROOT: undefined },
    );
    expect(exitCode).toBe(1);
    expect(stderr).toContain("Error: PARALLAX_REPO_ROOT is required");
  });

  it("US-001-AC01: subprocess (i2v.py) is spawned and exit code propagated (exit 0 from fake script)", async () => {
    const tmpRoot = await makeFakeLtx2I2vRoot("import sys; sys.exit(0)\n");
    try {
      const { exitCode } = await runCLIWithEnv(
        ["create", "video", "--model", "ltx2", "--prompt", "test", "--input", "/etc/hostname", "--models-dir", "/tmp"],
        { PARALLAX_REPO_ROOT: tmpRoot },
      );
      expect(exitCode).toBe(0);
    } finally {
      await rm(tmpRoot, { recursive: true, force: true });
    }
  });

  // AC02: --input is forwarded as --image (not --input) to the subprocess
  it("US-001-AC02: --input is forwarded as --image to the ltx2 i2v subprocess", async () => {
    const tmpRoot = await makeFakeLtx2I2vRoot(
      'import sys; print(" ".join(sys.argv[1:])); sys.exit(0)\n',
    );
    try {
      const { stdout, exitCode } = await runCLIWithEnv(
        ["create", "video", "--model", "ltx2", "--prompt", "test", "--input", "/etc/hostname", "--models-dir", "/tmp"],
        { PARALLAX_REPO_ROOT: tmpRoot },
      );
      expect(exitCode).toBe(0);
      expect(stdout).toContain("--image");
      expect(stdout).toContain("/etc/hostname");
      // bare --input must not be forwarded
      expect(stdout).not.toContain("--input");
    } finally {
      await rm(tmpRoot, { recursive: true, force: true });
    }
  });

  // AC03: --prompt, --width, --height, --length, --steps, --seed, --output are forwarded
  it("US-001-AC03: --prompt, --width, --height, --length, --steps, --seed, --output are forwarded", async () => {
    const tmpRoot = await makeFakeLtx2I2vRoot(
      'import sys; print(" ".join(sys.argv[1:])); sys.exit(0)\n',
    );
    try {
      const { stdout, exitCode } = await runCLIWithEnv(
        [
          "create", "video", "--model", "ltx2", "--prompt", "a cat video",
          "--input", "/etc/hostname", "--models-dir", "/tmp",
          "--width", "1280", "--height", "720", "--length", "97",
          "--steps", "20", "--seed", "42", "--output", "my_video.mp4",
        ],
        { PARALLAX_REPO_ROOT: tmpRoot },
      );
      expect(exitCode).toBe(0);
      expect(stdout).toContain("--prompt");
      expect(stdout).toContain("a cat video");
      expect(stdout).toContain("--width");
      expect(stdout).toContain("1280");
      expect(stdout).toContain("--height");
      expect(stdout).toContain("720");
      expect(stdout).toContain("--length");
      expect(stdout).toContain("97");
      expect(stdout).toContain("--steps");
      expect(stdout).toContain("20");
      expect(stdout).toContain("--seed");
      expect(stdout).toContain("42");
      expect(stdout).toContain("--output");
      expect(stdout).toContain("my_video.mp4");
    } finally {
      await rm(tmpRoot, { recursive: true, force: true });
    }
  });

  // AC04: --cfg is forwarded as --cfg-pass1, never bare --cfg
  it("US-001-AC04: --cfg is forwarded as --cfg-pass1 to the ltx2 i2v subprocess", async () => {
    const tmpRoot = await makeFakeLtx2I2vRoot(
      'import sys; print(" ".join(sys.argv[1:])); sys.exit(0)\n',
    );
    try {
      const { stdout, exitCode } = await runCLIWithEnv(
        ["create", "video", "--model", "ltx2", "--prompt", "test", "--input", "/etc/hostname", "--models-dir", "/tmp", "--cfg", "4.0"],
        { PARALLAX_REPO_ROOT: tmpRoot },
      );
      expect(exitCode).toBe(0);
      expect(stdout).toContain("--cfg-pass1");
      expect(stdout).toContain("4.0");
      expect(stdout).not.toContain("--cfg ");
    } finally {
      await rm(tmpRoot, { recursive: true, force: true });
    }
  });

  it("US-001-AC04: bare --cfg is NOT in i2v subprocess argv (exact element check)", async () => {
    const tmpRoot = await makeFakeLtx2I2vRoot(
      'import sys\nif "--cfg" in sys.argv[1:]:\n    sys.exit(2)\nsys.exit(0)\n',
    );
    try {
      const { exitCode } = await runCLIWithEnv(
        ["create", "video", "--model", "ltx2", "--prompt", "test", "--input", "/etc/hostname", "--models-dir", "/tmp", "--cfg", "4.0"],
        { PARALLAX_REPO_ROOT: tmpRoot },
      );
      // Would be 2 if bare --cfg were forwarded; 0 confirms it was converted to --cfg-pass1.
      expect(exitCode).toBe(0);
    } finally {
      await rm(tmpRoot, { recursive: true, force: true });
    }
  });

  // AC05: --models-dir is forwarded
  it("US-001-AC05: --models-dir is forwarded to the ltx2 i2v subprocess", async () => {
    const tmpRoot = await makeFakeLtx2I2vRoot(
      'import sys; print(" ".join(sys.argv[1:])); sys.exit(0)\n',
    );
    try {
      const { stdout, exitCode } = await runCLIWithEnv(
        ["create", "video", "--model", "ltx2", "--prompt", "test", "--input", "/etc/hostname", "--models-dir", "/my/models"],
        { PARALLAX_REPO_ROOT: tmpRoot },
      );
      expect(exitCode).toBe(0);
      expect(stdout).toContain("--models-dir");
      expect(stdout).toContain("/my/models");
    } finally {
      await rm(tmpRoot, { recursive: true, force: true });
    }
  });

  // AC06: exit code propagation
  it("US-001-AC06: CLI exits with non-zero when i2v subprocess fails (bad PARALLAX_REPO_ROOT path)", async () => {
    const { exitCode } = await runCLIWithEnv(
      ["create", "video", "--model", "ltx2", "--prompt", "test", "--input", "/etc/hostname", "--models-dir", "/tmp"],
      { PARALLAX_REPO_ROOT: "/nonexistent-parallax-root-12345" },
    );
    expect(exitCode).not.toBe(0);
  });

  it("US-001-AC06: CLI propagates i2v subprocess exit code 3 verbatim", async () => {
    const tmpRoot = await makeFakeLtx2I2vRoot("import sys; sys.exit(3)\n");
    try {
      const { exitCode } = await runCLIWithEnv(
        ["create", "video", "--model", "ltx2", "--prompt", "test", "--input", "/etc/hostname", "--models-dir", "/tmp"],
        { PARALLAX_REPO_ROOT: tmpRoot },
      );
      expect(exitCode).toBe(3);
    } finally {
      await rm(tmpRoot, { recursive: true, force: true });
    }
  });

  // Regression: t2v (no --input) still uses t2v.py for ltx2
  it("US-001-regression: ltx2 without --input still uses t2v.py (t2v mode)", async () => {
    const tmpRoot = await makeFakeLtx2Root("import sys; sys.exit(0)\n");
    try {
      const { exitCode } = await runCLIWithEnv(
        ["create", "video", "--model", "ltx2", "--prompt", "test", "--models-dir", "/tmp"],
        { PARALLAX_REPO_ROOT: tmpRoot },
      );
      expect(exitCode).toBe(0);
    } finally {
      await rm(tmpRoot, { recursive: true, force: true });
    }
  });

  // AC07: --input is listed in create video --help
  it("US-001-AC07: --input option is listed in create video --help", async () => {
    const { stdout, exitCode } = await runCLI(["create", "video", "--help"]);
    expect(exitCode).toBe(0);
    expect(stdout).toContain("--input");
  });
});

// Helper: create a temporary PARALLAX_REPO_ROOT with a fake ltx23 i2v.py script.
async function makeFakeLtx23I2vRoot(scriptBody: string): Promise<string> {
  const tmpRoot = await mkdtemp(join(tmpdir(), "ltx23_i2v_test_"));
  const scriptDir = join(tmpRoot, "examples", "video", "ltx", "ltx23");
  await mkdir(scriptDir, { recursive: true });
  await writeFile(join(scriptDir, "i2v.py"), scriptBody);
  return tmpRoot;
}

describe("parallax CLI — ltx23 i2v video generation (US-002)", () => {
  // AC01: running the command spawns uv run python examples/video/ltx/ltx23/i2v.py
  it("US-002-AC01: missing PYCOMFY_MODELS_DIR with --input exits 1 with clear error", async () => {
    const { stderr, exitCode } = await runCLIWithEnv(
      ["create", "video", "--model", "ltx23", "--prompt", "test", "--input", "/etc/hostname"],
      { PYCOMFY_MODELS_DIR: undefined, PARALLAX_REPO_ROOT: undefined },
    );
    expect(exitCode).toBe(1);
    expect(stderr).toContain("Error: --models-dir or PYCOMFY_MODELS_DIR is required");
  });

  it("US-002-AC01: with --input and --models-dir set, CLI reaches subprocess spawn (PARALLAX_REPO_ROOT check)", async () => {
    const { stderr, exitCode } = await runCLIWithEnv(
      ["create", "video", "--model", "ltx23", "--prompt", "test", "--input", "/etc/hostname", "--models-dir", "/tmp"],
      { PARALLAX_REPO_ROOT: undefined },
    );
    expect(exitCode).toBe(1);
    expect(stderr).toContain("Error: PARALLAX_REPO_ROOT is required");
  });

  it("US-002-AC01: subprocess (i2v.py) is spawned and exit code propagated (exit 0 from fake script)", async () => {
    const tmpRoot = await makeFakeLtx23I2vRoot("import sys; sys.exit(0)\n");
    try {
      const { exitCode } = await runCLIWithEnv(
        ["create", "video", "--model", "ltx23", "--prompt", "test", "--input", "/etc/hostname", "--models-dir", "/tmp"],
        { PARALLAX_REPO_ROOT: tmpRoot },
      );
      expect(exitCode).toBe(0);
    } finally {
      await rm(tmpRoot, { recursive: true, force: true });
    }
  });

  // AC02: --input is forwarded as --image (not --input)
  it("US-002-AC02: --input is forwarded as --image to the ltx23 i2v subprocess", async () => {
    const tmpRoot = await makeFakeLtx23I2vRoot(
      'import sys; print(" ".join(sys.argv[1:])); sys.exit(0)\n',
    );
    try {
      const { stdout, exitCode } = await runCLIWithEnv(
        ["create", "video", "--model", "ltx23", "--prompt", "test", "--input", "/etc/hostname", "--models-dir", "/tmp"],
        { PARALLAX_REPO_ROOT: tmpRoot },
      );
      expect(exitCode).toBe(0);
      expect(stdout).toContain("--image");
      expect(stdout).toContain("/etc/hostname");
      // bare --input must not be forwarded
      expect(stdout).not.toContain("--input");
    } finally {
      await rm(tmpRoot, { recursive: true, force: true });
    }
  });

  // AC03: --prompt, --width, --height, --length, --cfg, --seed, --output are forwarded
  it("US-002-AC03: --prompt, --width, --height, --length, --cfg, --seed, --output are forwarded", async () => {
    const tmpRoot = await makeFakeLtx23I2vRoot(
      'import sys; print(" ".join(sys.argv[1:])); sys.exit(0)\n',
    );
    try {
      const { stdout, exitCode } = await runCLIWithEnv(
        [
          "create", "video", "--model", "ltx23", "--prompt", "a scenic journey",
          "--input", "/etc/hostname", "--models-dir", "/tmp",
          "--width", "768", "--height", "512", "--length", "97",
          "--cfg", "1.0", "--seed", "42", "--output", "out.mp4",
        ],
        { PARALLAX_REPO_ROOT: tmpRoot },
      );
      expect(exitCode).toBe(0);
      expect(stdout).toContain("--prompt");
      expect(stdout).toContain("a scenic journey");
      expect(stdout).toContain("--width");
      expect(stdout).toContain("768");
      expect(stdout).toContain("--height");
      expect(stdout).toContain("512");
      expect(stdout).toContain("--length");
      expect(stdout).toContain("97");
      expect(stdout).toContain("--cfg");
      expect(stdout).toContain("1.0");
      expect(stdout).toContain("--seed");
      expect(stdout).toContain("42");
      expect(stdout).toContain("--output");
      expect(stdout).toContain("out.mp4");
    } finally {
      await rm(tmpRoot, { recursive: true, force: true });
    }
  });

  // AC04: --steps is NOT forwarded (ltx23 i2v is distilled)
  it("US-002-AC04: --steps is NOT forwarded to ltx23 i2v subprocess", async () => {
    const tmpRoot = await makeFakeLtx23I2vRoot(
      'import sys\nif "--steps" in sys.argv[1:]:\n    sys.exit(2)\nsys.exit(0)\n',
    );
    try {
      const { exitCode } = await runCLIWithEnv(
        ["create", "video", "--model", "ltx23", "--prompt", "test", "--input", "/etc/hostname", "--models-dir", "/tmp", "--steps", "20"],
        { PARALLAX_REPO_ROOT: tmpRoot },
      );
      // Would be 2 if --steps were forwarded; 0 confirms it was dropped.
      expect(exitCode).toBe(0);
    } finally {
      await rm(tmpRoot, { recursive: true, force: true });
    }
  });

  it("US-002-AC04: --steps is absent even when not explicitly passed", async () => {
    const tmpRoot = await makeFakeLtx23I2vRoot(
      'import sys\nif "--steps" in sys.argv[1:]:\n    sys.exit(2)\nsys.exit(0)\n',
    );
    try {
      const { exitCode } = await runCLIWithEnv(
        ["create", "video", "--model", "ltx23", "--prompt", "test", "--input", "/etc/hostname", "--models-dir", "/tmp"],
        { PARALLAX_REPO_ROOT: tmpRoot },
      );
      expect(exitCode).toBe(0);
    } finally {
      await rm(tmpRoot, { recursive: true, force: true });
    }
  });

  // AC05: --models-dir is forwarded
  it("US-002-AC05: --models-dir is forwarded to the ltx23 i2v subprocess", async () => {
    const tmpRoot = await makeFakeLtx23I2vRoot(
      'import sys; print(" ".join(sys.argv[1:])); sys.exit(0)\n',
    );
    try {
      const { stdout, exitCode } = await runCLIWithEnv(
        ["create", "video", "--model", "ltx23", "--prompt", "test", "--input", "/etc/hostname", "--models-dir", "/my/models"],
        { PARALLAX_REPO_ROOT: tmpRoot },
      );
      expect(exitCode).toBe(0);
      expect(stdout).toContain("--models-dir");
      expect(stdout).toContain("/my/models");
    } finally {
      await rm(tmpRoot, { recursive: true, force: true });
    }
  });

  // AC06: exit code propagation
  it("US-002-AC06: CLI exits with non-zero when i2v subprocess fails (bad PARALLAX_REPO_ROOT path)", async () => {
    const { exitCode } = await runCLIWithEnv(
      ["create", "video", "--model", "ltx23", "--prompt", "test", "--input", "/etc/hostname", "--models-dir", "/tmp"],
      { PARALLAX_REPO_ROOT: "/nonexistent-parallax-root-12345" },
    );
    expect(exitCode).not.toBe(0);
  });

  it("US-002-AC06: CLI propagates ltx23 i2v subprocess exit code 3 verbatim", async () => {
    const tmpRoot = await makeFakeLtx23I2vRoot("import sys; sys.exit(3)\n");
    try {
      const { exitCode } = await runCLIWithEnv(
        ["create", "video", "--model", "ltx23", "--prompt", "test", "--input", "/etc/hostname", "--models-dir", "/tmp"],
        { PARALLAX_REPO_ROOT: tmpRoot },
      );
      expect(exitCode).toBe(3);
    } finally {
      await rm(tmpRoot, { recursive: true, force: true });
    }
  });

  // Regression: ltx23 without --input still routes to t2v.py
  it("US-002-regression: ltx23 without --input still uses t2v.py (t2v mode)", async () => {
    const tmpRoot = await makeFakeLtx23Root("import sys; sys.exit(0)\n");
    try {
      const { exitCode } = await runCLIWithEnv(
        ["create", "video", "--model", "ltx23", "--prompt", "test", "--models-dir", "/tmp"],
        { PARALLAX_REPO_ROOT: tmpRoot },
      );
      expect(exitCode).toBe(0);
    } finally {
      await rm(tmpRoot, { recursive: true, force: true });
    }
  });
});

// Helper: create a temporary PARALLAX_REPO_ROOT with a fake wan21 i2v.py script.
async function makeFakeWan21I2vRoot(scriptBody: string): Promise<string> {
  const tmpRoot = await mkdtemp(join(tmpdir(), "wan21_i2v_test_"));
  const scriptDir = join(tmpRoot, "examples", "video", "wan", "wan21");
  await mkdir(scriptDir, { recursive: true });
  await writeFile(join(scriptDir, "i2v.py"), scriptBody);
  return tmpRoot;
}

describe("parallax CLI — wan21 i2v video generation (US-003-it39)", () => {
  // AC01: running the command spawns uv run python examples/video/wan/wan21/i2v.py
  it("US-003-AC01: missing PYCOMFY_MODELS_DIR with --input exits 1 with clear error", async () => {
    const { stderr, exitCode } = await runCLIWithEnv(
      ["create", "video", "--model", "wan21", "--prompt", "test", "--input", "/etc/hostname"],
      { PYCOMFY_MODELS_DIR: undefined, PARALLAX_REPO_ROOT: undefined },
    );
    expect(exitCode).toBe(1);
    expect(stderr).toContain("Error: --models-dir or PYCOMFY_MODELS_DIR is required");
  });

  it("US-003-AC01: with --input and --models-dir set, CLI reaches subprocess spawn (PARALLAX_REPO_ROOT check)", async () => {
    const { stderr, exitCode } = await runCLIWithEnv(
      ["create", "video", "--model", "wan21", "--prompt", "test", "--input", "/etc/hostname", "--models-dir", "/tmp"],
      { PARALLAX_REPO_ROOT: undefined },
    );
    expect(exitCode).toBe(1);
    expect(stderr).toContain("Error: PARALLAX_REPO_ROOT is required");
  });

  it("US-003-AC01: subprocess (i2v.py) is spawned and exit code propagated (exit 0 from fake script)", async () => {
    const tmpRoot = await makeFakeWan21I2vRoot("import sys; sys.exit(0)\n");
    try {
      const { exitCode } = await runCLIWithEnv(
        ["create", "video", "--model", "wan21", "--prompt", "test", "--input", "/etc/hostname", "--models-dir", "/tmp"],
        { PARALLAX_REPO_ROOT: tmpRoot },
      );
      expect(exitCode).toBe(0);
    } finally {
      await rm(tmpRoot, { recursive: true, force: true });
    }
  });

  // AC02: --input <path> is forwarded as --image <path>
  it("US-003-AC02: --input is forwarded as --image to the wan21 i2v subprocess", async () => {
    const tmpRoot = await makeFakeWan21I2vRoot(
      'import sys; print(" ".join(sys.argv[1:])); sys.exit(0)\n',
    );
    try {
      const { stdout, exitCode } = await runCLIWithEnv(
        ["create", "video", "--model", "wan21", "--prompt", "test", "--input", "/etc/hostname", "--models-dir", "/tmp"],
        { PARALLAX_REPO_ROOT: tmpRoot },
      );
      expect(exitCode).toBe(0);
      expect(stdout).toContain("--image");
      expect(stdout).toContain("/etc/hostname");
      // bare --input must not be forwarded to the subprocess
      expect(stdout).not.toContain("--input");
    } finally {
      await rm(tmpRoot, { recursive: true, force: true });
    }
  });

  // AC03: --prompt, --width, --height, --length, --steps, --cfg, --seed are forwarded
  it("US-003-AC03: --prompt, --width, --height, --length, --steps, --cfg, --seed, --output are forwarded", async () => {
    const tmpRoot = await makeFakeWan21I2vRoot(
      'import sys; print(" ".join(sys.argv[1:])); sys.exit(0)\n',
    );
    try {
      const { stdout, exitCode } = await runCLIWithEnv(
        [
          "create", "video", "--model", "wan21", "--prompt", "a cat in the snow",
          "--input", "/etc/hostname", "--models-dir", "/tmp",
          "--width", "512", "--height", "512", "--length", "33",
          "--steps", "20", "--cfg", "6", "--seed", "42", "--output", "wan21_i2v.mp4",
        ],
        { PARALLAX_REPO_ROOT: tmpRoot },
      );
      expect(exitCode).toBe(0);
      expect(stdout).toContain("--prompt");
      expect(stdout).toContain("a cat in the snow");
      expect(stdout).toContain("--width");
      expect(stdout).toContain("512");
      expect(stdout).toContain("--height");
      expect(stdout).toContain("512");
      expect(stdout).toContain("--length");
      expect(stdout).toContain("33");
      expect(stdout).toContain("--steps");
      expect(stdout).toContain("20");
      expect(stdout).toContain("--cfg");
      expect(stdout).toContain("6");
      expect(stdout).toContain("--seed");
      expect(stdout).toContain("42");
      expect(stdout).toContain("--output");
      expect(stdout).toContain("wan21_i2v.mp4");
    } finally {
      await rm(tmpRoot, { recursive: true, force: true });
    }
  });

  // AC04: --models-dir is forwarded
  it("US-003-AC04: --models-dir is forwarded to the wan21 i2v subprocess", async () => {
    const tmpRoot = await makeFakeWan21I2vRoot(
      'import sys; print(" ".join(sys.argv[1:])); sys.exit(0)\n',
    );
    try {
      const { stdout, exitCode } = await runCLIWithEnv(
        ["create", "video", "--model", "wan21", "--prompt", "test", "--input", "/etc/hostname", "--models-dir", "/my/models"],
        { PARALLAX_REPO_ROOT: tmpRoot },
      );
      expect(exitCode).toBe(0);
      expect(stdout).toContain("--models-dir");
      expect(stdout).toContain("/my/models");
    } finally {
      await rm(tmpRoot, { recursive: true, force: true });
    }
  });

  // AC05: CLI exits with the subprocess exit code
  it("US-003-AC05: CLI exits with non-zero when i2v subprocess fails (bad PARALLAX_REPO_ROOT path)", async () => {
    const { exitCode } = await runCLIWithEnv(
      ["create", "video", "--model", "wan21", "--prompt", "test", "--input", "/etc/hostname", "--models-dir", "/tmp"],
      { PARALLAX_REPO_ROOT: "/nonexistent-parallax-root-12345" },
    );
    expect(exitCode).not.toBe(0);
  });

  it("US-003-AC05: CLI propagates wan21 i2v subprocess exit code 3 verbatim", async () => {
    const tmpRoot = await makeFakeWan21I2vRoot("import sys; sys.exit(3)\n");
    try {
      const { exitCode } = await runCLIWithEnv(
        ["create", "video", "--model", "wan21", "--prompt", "test", "--input", "/etc/hostname", "--models-dir", "/tmp"],
        { PARALLAX_REPO_ROOT: tmpRoot },
      );
      expect(exitCode).toBe(3);
    } finally {
      await rm(tmpRoot, { recursive: true, force: true });
    }
  });

  // AC06: typecheck / lint passes (validated by running `bun run tsc --noEmit`)
  it("US-003-AC06: --input option is listed in create video --help", async () => {
    const { stdout, exitCode } = await runCLI(["create", "video", "--help"]);
    expect(exitCode).toBe(0);
    expect(stdout).toContain("--input");
    expect(stdout).toContain("wan21");
  });

  // Regression: wan21 without --input still routes to t2v.py
  it("US-003-regression: wan21 without --input still uses t2v.py (t2v mode)", async () => {
    const tmpRoot = await makeFakeWan21Root("import sys; sys.exit(0)\n");
    try {
      const { exitCode } = await runCLIWithEnv(
        ["create", "video", "--model", "wan21", "--prompt", "test", "--models-dir", "/tmp"],
        { PARALLAX_REPO_ROOT: tmpRoot },
      );
      expect(exitCode).toBe(0);
    } finally {
      await rm(tmpRoot, { recursive: true, force: true });
    }
  });
});

describe("parallax CLI — wan22 i2v video generation (US-004-it39)", () => {
  // AC01: running the command spawns uv run python examples/video/wan/wan22/i2v.py
  it("US-004-AC01: missing PYCOMFY_MODELS_DIR with --input exits 1 with clear error", async () => {
    const { stderr, exitCode } = await runCLIWithEnv(
      ["create", "video", "--model", "wan22", "--prompt", "test", "--input", "/etc/hostname"],
      { PYCOMFY_MODELS_DIR: undefined, PARALLAX_REPO_ROOT: undefined },
    );
    expect(exitCode).toBe(1);
    expect(stderr).toContain("Error: --models-dir or PYCOMFY_MODELS_DIR is required");
  });

  it("US-004-AC01: with --input and --models-dir set, CLI reaches subprocess spawn (PARALLAX_REPO_ROOT check)", async () => {
    const { stderr, exitCode } = await runCLIWithEnv(
      ["create", "video", "--model", "wan22", "--prompt", "test", "--input", "/etc/hostname", "--models-dir", "/tmp"],
      { PARALLAX_REPO_ROOT: undefined },
    );
    expect(exitCode).toBe(1);
    expect(stderr).toContain("Error: PARALLAX_REPO_ROOT is required");
  });

  it("US-004-AC01: subprocess (i2v.py) is spawned and exit code propagated (exit 0 from fake script)", async () => {
    const tmpRoot = await makeFakeWan22I2vRoot("import sys; sys.exit(0)\n");
    try {
      const { exitCode } = await runCLIWithEnv(
        ["create", "video", "--model", "wan22", "--prompt", "test", "--input", "/etc/hostname", "--models-dir", "/tmp"],
        { PARALLAX_REPO_ROOT: tmpRoot },
      );
      expect(exitCode).toBe(0);
    } finally {
      await rm(tmpRoot, { recursive: true, force: true });
    }
  });

  // AC02: --input <path> is forwarded as --image <path>
  it("US-004-AC02: --input is forwarded as --image to the wan22 i2v subprocess", async () => {
    const tmpRoot = await makeFakeWan22I2vRoot(
      'import sys; print(" ".join(sys.argv[1:])); sys.exit(0)\n',
    );
    try {
      const { stdout, exitCode } = await runCLIWithEnv(
        ["create", "video", "--model", "wan22", "--prompt", "test", "--input", "/etc/hostname", "--models-dir", "/tmp"],
        { PARALLAX_REPO_ROOT: tmpRoot },
      );
      expect(exitCode).toBe(0);
      expect(stdout).toContain("--image");
      expect(stdout).toContain("/etc/hostname");
      expect(stdout).not.toContain("--input");
    } finally {
      await rm(tmpRoot, { recursive: true, force: true });
    }
  });

  // AC03: --prompt, --width, --height, --length, --steps, --cfg, --seed are forwarded
  it("US-004-AC03: --prompt, --width, --height, --length, --steps, --cfg, --seed, --output are forwarded", async () => {
    const tmpRoot = await makeFakeWan22I2vRoot(
      'import sys; print(" ".join(sys.argv[1:])); sys.exit(0)\n',
    );
    try {
      const { stdout, exitCode } = await runCLIWithEnv(
        [
          "create", "video", "--model", "wan22", "--prompt", "a dragon over mountains",
          "--input", "/etc/hostname", "--models-dir", "/tmp",
          "--width", "640", "--height", "640", "--length", "81",
          "--steps", "20", "--cfg", "3.5", "--seed", "77", "--output", "wan22_i2v.mp4",
        ],
        { PARALLAX_REPO_ROOT: tmpRoot },
      );
      expect(exitCode).toBe(0);
      expect(stdout).toContain("--prompt");
      expect(stdout).toContain("a dragon over mountains");
      expect(stdout).toContain("--width");
      expect(stdout).toContain("640");
      expect(stdout).toContain("--height");
      expect(stdout).toContain("640");
      expect(stdout).toContain("--length");
      expect(stdout).toContain("81");
      expect(stdout).toContain("--steps");
      expect(stdout).toContain("20");
      expect(stdout).toContain("--cfg");
      expect(stdout).toContain("3.5");
      expect(stdout).toContain("--seed");
      expect(stdout).toContain("77");
      expect(stdout).toContain("--output");
      expect(stdout).toContain("wan22_i2v.mp4");
    } finally {
      await rm(tmpRoot, { recursive: true, force: true });
    }
  });

  // AC04: --models-dir is forwarded
  it("US-004-AC04: --models-dir is forwarded to the wan22 i2v subprocess", async () => {
    const tmpRoot = await makeFakeWan22I2vRoot(
      'import sys; print(" ".join(sys.argv[1:])); sys.exit(0)\n',
    );
    try {
      const { stdout, exitCode } = await runCLIWithEnv(
        ["create", "video", "--model", "wan22", "--prompt", "test", "--input", "/etc/hostname", "--models-dir", "/my/wan22/models"],
        { PARALLAX_REPO_ROOT: tmpRoot },
      );
      expect(exitCode).toBe(0);
      expect(stdout).toContain("--models-dir");
      expect(stdout).toContain("/my/wan22/models");
    } finally {
      await rm(tmpRoot, { recursive: true, force: true });
    }
  });

  // AC05: CLI exits with the subprocess exit code
  it("US-004-AC05: CLI exits with non-zero when i2v subprocess fails (bad PARALLAX_REPO_ROOT path)", async () => {
    const { exitCode } = await runCLIWithEnv(
      ["create", "video", "--model", "wan22", "--prompt", "test", "--input", "/etc/hostname", "--models-dir", "/tmp"],
      { PARALLAX_REPO_ROOT: "/nonexistent-parallax-root-12345" },
    );
    expect(exitCode).not.toBe(0);
  });

  it("US-004-AC05: CLI propagates wan22 i2v subprocess exit code 3 verbatim", async () => {
    const tmpRoot = await makeFakeWan22I2vRoot("import sys; sys.exit(3)\n");
    try {
      const { exitCode } = await runCLIWithEnv(
        ["create", "video", "--model", "wan22", "--prompt", "test", "--input", "/etc/hostname", "--models-dir", "/tmp"],
        { PARALLAX_REPO_ROOT: tmpRoot },
      );
      expect(exitCode).toBe(3);
    } finally {
      await rm(tmpRoot, { recursive: true, force: true });
    }
  });

  // AC06: typecheck / lint passes — verify --input help mentions wan22
  it("US-004-AC06: --input option in create video --help lists wan22", async () => {
    const { stdout, exitCode } = await runCLI(["create", "video", "--help"]);
    expect(exitCode).toBe(0);
    expect(stdout).toContain("--input");
    expect(stdout).toContain("wan22");
  });

  // Regression: wan22 without --input still routes to t2v.py
  it("US-004-regression: wan22 without --input still uses t2v.py (t2v mode)", async () => {
    const tmpRoot = await makeFakeWan22Root("import sys; sys.exit(0)\n");
    try {
      const { exitCode } = await runCLIWithEnv(
        ["create", "video", "--model", "wan22", "--prompt", "test", "--models-dir", "/tmp"],
        { PARALLAX_REPO_ROOT: tmpRoot },
      );
      expect(exitCode).toBe(0);
    } finally {
      await rm(tmpRoot, { recursive: true, force: true });
    }
  });
});

describe("parallax CLI — --input flag on create video (US-005-it39)", () => {
  // AC01: --input is listed in create video --help
  it("US-005-AC01: create video --help lists --input as an optional flag", async () => {
    const { stdout, exitCode } = await runCLI(["create", "video", "--help"]);
    expect(exitCode).toBe(0);
    expect(stdout).toContain("--input");
  });

  // AC02: --input present → routes to i2v script (using wan21 as representative model)
  it("US-005-AC02: --input routes to i2v.py (wan21)", async () => {
    const tmpRoot = await mkdtemp(join(tmpdir(), "us005_i2v_test_"));
    const scriptDir = join(tmpRoot, "examples", "video", "wan", "wan21");
    await mkdir(scriptDir, { recursive: true });
    // create a real fake input image file so existence check passes
    const fakeInput = join(tmpRoot, "input.png");
    await writeFile(fakeInput, "fake");
    await writeFile(join(scriptDir, "i2v.py"), 'import sys; print("i2v"); sys.exit(0)\n');
    try {
      const { stdout, exitCode } = await runCLIWithEnv(
        ["create", "video", "--model", "wan21", "--prompt", "test", "--input", fakeInput, "--models-dir", "/tmp"],
        { PARALLAX_REPO_ROOT: tmpRoot },
      );
      expect(exitCode).toBe(0);
      expect(stdout).toContain("i2v");
    } finally {
      await rm(tmpRoot, { recursive: true, force: true });
    }
  });

  // AC03: --input absent → routes to t2v script (no regression)
  it("US-005-AC03: without --input, routes to t2v.py (wan21, no regression)", async () => {
    const tmpRoot = await mkdtemp(join(tmpdir(), "us005_t2v_test_"));
    const scriptDir = join(tmpRoot, "examples", "video", "wan", "wan21");
    await mkdir(scriptDir, { recursive: true });
    await writeFile(join(scriptDir, "t2v.py"), 'import sys; print("t2v"); sys.exit(0)\n');
    try {
      const { stdout, exitCode } = await runCLIWithEnv(
        ["create", "video", "--model", "wan21", "--prompt", "test", "--models-dir", "/tmp"],
        { PARALLAX_REPO_ROOT: tmpRoot },
      );
      expect(exitCode).toBe(0);
      expect(stdout).toContain("t2v");
    } finally {
      await rm(tmpRoot, { recursive: true, force: true });
    }
  });

  // AC04: --input provided but file does not exist → error + exit 1
  it("US-005-AC04: --input with non-existent file prints error and exits 1", async () => {
    const { stderr, exitCode } = await runCLIWithEnv(
      ["create", "video", "--model", "wan21", "--prompt", "test", "--input", "/nonexistent/missing_image.png", "--models-dir", "/tmp"],
      { PARALLAX_REPO_ROOT: "/tmp" },
    );
    expect(exitCode).toBe(1);
    expect(stderr).toContain("Error: input file not found:");
    expect(stderr).toContain("/nonexistent/missing_image.png");
  });

  it("US-005-AC04: no subprocess is spawned when input file is missing (PARALLAX_REPO_ROOT unset)", async () => {
    // If the file-not-found check fires before subprocess spawn, exit code is 1
    // even when PARALLAX_REPO_ROOT is unset (spawn would have exited 1 too, but for a different reason)
    const { stderr, exitCode } = await runCLIWithEnv(
      ["create", "video", "--model", "ltx2", "--prompt", "test", "--input", "/definitely/not/here.jpg", "--models-dir", "/tmp"],
      { PARALLAX_REPO_ROOT: undefined },
    );
    expect(exitCode).toBe(1);
    // must be the input-file error, NOT the PARALLAX_REPO_ROOT error
    expect(stderr).toContain("Error: input file not found:");
    expect(stderr).not.toContain("PARALLAX_REPO_ROOT");
  });
});
