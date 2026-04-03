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
    const { stderr, exitCode } = await runCLI(["edit", "image", "--model", "badmodel", "--prompt", "test", "--input", "img.png"]);
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
    const { stderr, exitCode } = await runCLI(["edit", "image", "--prompt", "test", "--input", "img.png"]);
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
    const { stderr, exitCode } = await runCLI(["edit", "image", "--model", "qwen", "--input", "img.png"]);
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
  it("US-007-AC01/02: create video with valid flags prints stub message and exits 0", async () => {
    const { stdout, exitCode } = await runCLI(["create", "video", "--model", "wan21", "--prompt", "test"]);
    expect(exitCode).toBe(0);
    expect(stdout).toContain("[parallax] create video --model wan21 — not yet implemented (coming soon)");
  });

  it("US-007-AC01/02: create audio with valid flags prints stub message and exits 0", async () => {
    const { stdout, exitCode } = await runCLI(["create", "audio", "--model", "ace_step", "--prompt", "test"]);
    expect(exitCode).toBe(0);
    expect(stdout).toContain("[parallax] create audio --model ace_step — not yet implemented (coming soon)");
  });

  it("US-007-AC01/02: edit image with valid flags prints stub message and exits 0", async () => {
    const { stdout, exitCode } = await runCLI(["edit", "image", "--model", "qwen", "--prompt", "test", "--input", "img.png"]);
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
