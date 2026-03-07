import { describe, expect, test } from "bun:test";

import {
  buildCommand,
  ensureAgentCommandAvailable,
  invokeAgent,
  parseAgentArg,
  parseProvider,
} from "./agent";

describe("agent provider parsing", () => {
  test("accepts copilot as a valid provider and maps to copilot CLI defaults", () => {
    const parsed = parseAgentArg(["create", "--agent", "copilot", "--force"]);

    expect(parsed.provider).toBe("copilot");
    expect(parsed.remainingArgs).toEqual(["create", "--force"]);
    expect(parseProvider("copilot")).toBe("copilot");
    expect(buildCommand("copilot")).toEqual({ cmd: "copilot", args: ["-p", "--yolo"] });
  });

  test("accepts cursor as a valid provider in --agent argument parsing", () => {
    const parsed = parseAgentArg(["create", "--agent", "cursor", "--force"]);

    expect(parsed.provider).toBe("cursor");
    expect(parsed.remainingArgs).toEqual(["create", "--force"]);
  });

  test("maps cursor provider to the Cursor agent CLI binary", () => {
    expect(parseProvider("cursor")).toBe("cursor");
    expect(buildCommand("cursor")).toEqual({ cmd: "agent", args: [] });
  });

  test("unknown provider error includes cursor in valid provider list", () => {
    expect(() => parseProvider("unknown-provider")).toThrow(
      "Unknown agent provider 'unknown-provider'. Valid providers: claude, codex, gemini, cursor, copilot",
    );
  });
});

describe("agent invocation command availability", () => {
  test("throws when copilot provider is selected but `copilot` is not in PATH", () => {
    expect(() =>
      ensureAgentCommandAvailable("copilot", () => null),
    ).toThrow("Required CLI 'copilot' for provider 'copilot' is not in PATH.");
  });

  test("returns a clear error when cursor provider is selected but `agent` is not in PATH", async () => {
    await expect(
      invokeAgent({
        provider: "cursor",
        prompt: "Test prompt",
        resolveCommandPath: () => null,
      }),
    ).rejects.toThrow(
      "Cursor agent CLI is unavailable: `agent` command not found in PATH. Install/configure Cursor Agent CLI or use another provider.",
    );
  });

  test("invokes copilot interactive mode with -i and inherited stdio", async () => {
    const originalSpawn = Bun.spawn;
    let capturedCmd: string[] | undefined;
    let capturedStdio:
      | {
          stdin: "ignore" | "inherit";
          stdout: "inherit" | "pipe";
          stderr: "inherit" | "pipe";
        }
      | undefined;

    try {
      (Bun as { spawn: typeof Bun.spawn }).spawn = ((cmd, options) => {
        const spawnOptions = options ?? {};
        capturedCmd = cmd as string[];
        capturedStdio = {
          stdin: spawnOptions.stdin as "ignore" | "inherit",
          stdout: spawnOptions.stdout as "inherit" | "pipe",
          stderr: spawnOptions.stderr as "inherit" | "pipe",
        };
        return {
          exited: Promise.resolve(0),
        } as unknown as ReturnType<typeof Bun.spawn>;
      }) as typeof Bun.spawn;

      const result = await invokeAgent({
        provider: "copilot",
        prompt: "Interview me",
        interactive: true,
        resolveCommandPath: () => "/usr/bin/copilot",
      });

      expect(capturedCmd).toEqual(["copilot", "-i", "Interview me"]);
      expect(capturedStdio).toEqual({
        stdin: "inherit",
        stdout: "inherit",
        stderr: "inherit",
      });
      expect(result).toEqual({ exitCode: 0, stdout: "", stderr: "" });
    } finally {
      (Bun as { spawn: typeof Bun.spawn }).spawn = originalSpawn;
    }
  });

  test("invokes copilot non-interactive mode with -p <prompt> --yolo and captures output", async () => {
    const originalSpawn = Bun.spawn;
    let capturedCmd: string[] | undefined;
    let capturedStdio:
      | {
          stdin: "ignore" | "inherit";
          stdout: "inherit" | "pipe";
          stderr: "inherit" | "pipe";
        }
      | undefined;

    const createStream = (text: string): ReadableStream<Uint8Array> => {
      const encoded = new TextEncoder().encode(text);
      return new ReadableStream<Uint8Array>({
        start(controller) {
          controller.enqueue(encoded);
          controller.close();
        },
      });
    };

    try {
      (Bun as { spawn: typeof Bun.spawn }).spawn = ((cmd, options) => {
        const spawnOptions = options ?? {};
        capturedCmd = cmd as string[];
        capturedStdio = {
          stdin: spawnOptions.stdin as "ignore" | "inherit",
          stdout: spawnOptions.stdout as "inherit" | "pipe",
          stderr: spawnOptions.stderr as "inherit" | "pipe",
        };
        return {
          exited: Promise.resolve(0),
          stdout: createStream("ok"),
          stderr: createStream(""),
        } as unknown as ReturnType<typeof Bun.spawn>;
      }) as typeof Bun.spawn;

      const result = await invokeAgent({
        provider: "copilot",
        prompt: "Do work",
        interactive: false,
        resolveCommandPath: () => "/usr/bin/copilot",
      });

      expect(capturedCmd).toEqual(["copilot", "-p", "Do work", "--yolo"]);
      expect(capturedStdio).toEqual({
        stdin: "inherit",
        stdout: "pipe",
        stderr: "pipe",
      });
      expect(result).toEqual({ exitCode: 0, stdout: "ok", stderr: "" });
    } finally {
      (Bun as { spawn: typeof Bun.spawn }).spawn = originalSpawn;
    }
  });
});
