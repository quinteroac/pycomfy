/**
 * Tests for US-003 — Harden the interface against edge cases
 */
import { render, screen, act, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { App } from "../App";
import { ChatBubble } from "../components/ChatBubble";
import type { ChatMessage } from "../types";

// ── EventSource mock (shared with ChatStream.test.tsx pattern) ────────────────

type ESListener = (event: MessageEvent) => void;
type ErrorListener = (event: Event) => void;

class MockEventSource {
  static instances: MockEventSource[] = [];

  url: string;
  onmessage: ESListener | null = null;
  onerror: ErrorListener | null = null;

  constructor(url: string) {
    this.url = url;
    MockEventSource.instances.push(this);
  }

  emit(data: object) {
    this.onmessage?.({ data: JSON.stringify(data) } as MessageEvent);
  }

  close() {}
}

// ── Setup / Teardown ──────────────────────────────────────────────────────────

beforeEach(() => {
  MockEventSource.instances = [];

  Object.defineProperty(globalThis, "URL", {
    value: {
      createObjectURL: vi.fn(() => "blob:mock-url"),
      revokeObjectURL: vi.fn(),
    },
    writable: true,
    configurable: true,
  });

  vi.stubGlobal("EventSource", MockEventSource);
});

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

function mockFetchWithJobId(jobId = "job-harden") {
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ job_id: jobId }),
    })
  );
}

// ── AC01: Long prompts don't break layout ─────────────────────────────────────

describe("US-003-AC01 — long prompt handling", () => {
  it("renders user bubble with 501-char content without error", () => {
    const longContent = "a".repeat(501);
    const msg: ChatMessage = {
      id: "m1",
      role: "user",
      content: longContent,
      status: "complete",
    };
    render(<ChatBubble message={msg} />);
    const bubble = screen.getByTestId("bubble-user");
    expect(bubble).toBeInTheDocument();
    expect(bubble).toHaveTextContent(longContent);
  });

  it("renders assistant bubble with 600-char content without error", () => {
    const longContent = "b".repeat(600);
    const msg: ChatMessage = {
      id: "m2",
      role: "assistant",
      content: longContent,
      status: "complete",
    };
    render(<ChatBubble message={msg} />);
    const bubble = screen.getByTestId("bubble-assistant");
    expect(bubble).toBeInTheDocument();
    expect(bubble).toHaveTextContent(longContent);
  });

  it("renders user bubble with long URL-like string (no spaces) without overflow", () => {
    const noSpaceContent = "x".repeat(300) + "/" + "y".repeat(300);
    const msg: ChatMessage = {
      id: "m3",
      role: "user",
      content: noSpaceContent,
      status: "complete",
    };
    render(<ChatBubble message={msg} />);
    expect(screen.getByTestId("bubble-user")).toBeInTheDocument();
  });
});

// ── AC02: SSE timeout warning after 30s ───────────────────────────────────────

describe("US-003-AC02 — SSE 30s timeout", () => {
  it("shows timeout warning when no SSE events arrive for 30 seconds", async () => {
    mockFetchWithJobId("job-timeout");

    // Intercept the 30s timeout so we can trigger it manually in the test
    let capturedTimeoutFn: (() => void) | null = null;
    const originalSetTimeout = globalThis.setTimeout;
    const originalClearTimeout = globalThis.clearTimeout;

    vi.spyOn(globalThis, "setTimeout").mockImplementation(
      (fn: TimerHandler, delay?: number, ...args: unknown[]) => {
        if (delay === 30_000) {
          capturedTimeoutFn = fn as () => void;
          return 99999 as unknown as ReturnType<typeof setTimeout>;
        }
        return originalSetTimeout.call(
          globalThis,
          fn as TimerHandler,
          delay,
          ...args
        );
      }
    );
    vi.spyOn(globalThis, "clearTimeout").mockImplementation((id) => {
      if (id === 99999) return;
      originalClearTimeout.call(
        globalThis,
        id as ReturnType<typeof setTimeout>
      );
    });

    const user = userEvent.setup();
    render(<App />);

    await user.type(screen.getByTestId("prompt-input"), "test");
    await user.click(screen.getByRole("button", { name: "Generate" }));

    await waitFor(() =>
      expect(MockEventSource.instances.length).toBeGreaterThan(0)
    );

    expect(capturedTimeoutFn).not.toBeNull();

    act(() => {
      capturedTimeoutFn!();
    });

    await waitFor(() => {
      expect(screen.getByTestId("bubble-assistant")).toHaveTextContent(
        /stalled/i
      );
    });
  });

  it("timeout warning message mentions 30 seconds", async () => {
    mockFetchWithJobId("job-timeout-msg");

    let capturedTimeoutFn: (() => void) | null = null;
    const originalSetTimeout = globalThis.setTimeout;
    const originalClearTimeout = globalThis.clearTimeout;

    vi.spyOn(globalThis, "setTimeout").mockImplementation(
      (fn: TimerHandler, delay?: number, ...args: unknown[]) => {
        if (delay === 30_000) {
          capturedTimeoutFn = fn as () => void;
          return 99999 as unknown as ReturnType<typeof setTimeout>;
        }
        return originalSetTimeout.call(
          globalThis,
          fn as TimerHandler,
          delay,
          ...args
        );
      }
    );
    vi.spyOn(globalThis, "clearTimeout").mockImplementation((id) => {
      if (id === 99999) return;
      originalClearTimeout.call(
        globalThis,
        id as ReturnType<typeof setTimeout>
      );
    });

    const user = userEvent.setup();
    render(<App />);

    await user.type(screen.getByTestId("prompt-input"), "test");
    await user.click(screen.getByRole("button", { name: "Generate" }));

    await waitFor(() =>
      expect(MockEventSource.instances.length).toBeGreaterThan(0)
    );

    act(() => {
      capturedTimeoutFn!();
    });

    await waitFor(() => {
      expect(screen.getByTestId("bubble-assistant")).toHaveTextContent(
        /30 seconds/i
      );
    });
  });

  it("sets up a 30-second timeout when EventSource is created", async () => {
    mockFetchWithJobId("job-timeout-setup");

    const originalSetTimeout = globalThis.setTimeout;
    const timeoutDelays: number[] = [];

    vi.spyOn(globalThis, "setTimeout").mockImplementation(
      (fn: TimerHandler, delay?: number, ...args: unknown[]) => {
        if (delay !== undefined) timeoutDelays.push(delay);
        return originalSetTimeout.call(
          globalThis,
          fn as TimerHandler,
          delay,
          ...args
        );
      }
    );

    const user = userEvent.setup();
    render(<App />);

    await user.type(screen.getByTestId("prompt-input"), "test");
    await user.click(screen.getByRole("button", { name: "Generate" }));

    await waitFor(() =>
      expect(MockEventSource.instances.length).toBeGreaterThan(0)
    );

    expect(timeoutDelays).toContain(30_000);
  });
});

// ── AC03: Empty state placeholder ────────────────────────────────────────────

describe("US-003-AC03 — empty state placeholder", () => {
  it("shows empty state when no messages", () => {
    render(<App />);
    expect(screen.getByTestId("chat-empty")).toBeInTheDocument();
  });

  it("empty state contains helpful instructions for the user", () => {
    render(<App />);
    const empty = screen.getByTestId("chat-empty");
    expect(empty).toHaveTextContent(/prompt/i);
    expect(empty).toHaveTextContent(/generate/i);
  });

  it("empty state is replaced by chat messages after first submission", async () => {
    mockFetchWithJobId();
    const user = userEvent.setup();
    render(<App />);

    expect(screen.getByTestId("chat-empty")).toBeInTheDocument();

    await user.type(screen.getByTestId("prompt-input"), "hello");
    await user.click(screen.getByRole("button", { name: "Generate" }));

    await waitFor(() => {
      expect(screen.queryByTestId("chat-empty")).not.toBeInTheDocument();
    });
    expect(screen.getByTestId("chat-messages")).toBeInTheDocument();
  });
});

// ── AC04: Visible focus states ────────────────────────────────────────────────

describe("US-003-AC04 — keyboard focus states", () => {
  it("prompt textarea can receive focus", () => {
    render(<App />);
    const textarea = screen.getByTestId("prompt-input");
    textarea.focus();
    expect(document.activeElement).toBe(textarea);
  });

  it("Generate button can receive focus", () => {
    render(<App />);
    const btn = screen.getByRole("button", { name: "Generate" });
    btn.focus();
    expect(document.activeElement).toBe(btn);
  });

  it("media type buttons can receive focus", () => {
    render(<App />);
    const imagebtn = screen.getByTestId("media-type-image");
    imagebtn.focus();
    expect(document.activeElement).toBe(imagebtn);
  });

  it("Generate button has type=button (no accidental form submission)", () => {
    render(<App />);
    const btn = screen.getByRole("button", { name: "Generate" });
    expect(btn).toHaveAttribute("type", "button");
  });
});

// ── AC05: Server error messages displayed in full ─────────────────────────────

describe("US-003-AC05 — server error messages", () => {
  it("displays the full server error response body in the assistant bubble", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: false,
        status: 422,
        text: async () => "Model not found: sdxl-turbo",
      })
    );

    const user = userEvent.setup();
    render(<App />);

    await user.type(screen.getByTestId("prompt-input"), "test");
    await user.click(screen.getByRole("button", { name: "Generate" }));

    await waitFor(() => {
      const bubble = screen.getByTestId("bubble-assistant");
      expect(bubble).toHaveTextContent("422");
      expect(bubble).toHaveTextContent("Model not found: sdxl-turbo");
    });
  });

  it("displays HTTP status when response body is empty", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: false,
        status: 500,
        text: async () => "",
      })
    );

    const user = userEvent.setup();
    render(<App />);

    await user.type(screen.getByTestId("prompt-input"), "test");
    await user.click(screen.getByRole("button", { name: "Generate" }));

    await waitFor(() => {
      expect(screen.getByTestId("bubble-assistant")).toHaveTextContent(
        "Server error 500"
      );
    });
  });

  it("displays SSE error field content in full", async () => {
    mockFetchWithJobId("job-sse-err");
    const user = userEvent.setup();
    render(<App />);

    await user.type(screen.getByTestId("prompt-input"), "test");
    await user.click(screen.getByRole("button", { name: "Generate" }));

    await waitFor(() =>
      expect(MockEventSource.instances.length).toBeGreaterThan(0)
    );
    const es = MockEventSource.instances[MockEventSource.instances.length - 1];

    act(() => {
      es.emit({
        step: "error",
        pct: 0,
        error: "CUDA out of memory. Tried to allocate 4.00 GiB.",
      });
    });

    await waitFor(() => {
      expect(screen.getByTestId("bubble-assistant")).toHaveTextContent(
        "CUDA out of memory. Tried to allocate 4.00 GiB."
      );
    });
  });
});
