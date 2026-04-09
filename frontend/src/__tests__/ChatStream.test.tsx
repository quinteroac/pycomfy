/**
 * Tests for US-003 — Submit prompt and stream generation progress
 */
import { render, screen, act, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { App } from "../App";

// ── EventSource mock ──────────────────────────────────────────────────────────

type ESListener = (event: MessageEvent) => void;
type ErrorListener = (event: Event) => void;

class MockEventSource {
  static instances: MockEventSource[] = [];

  url: string;
  onmessage: ESListener | null = null;
  onerror: ErrorListener | null = null;
  private _closed = false;

  constructor(url: string) {
    this.url = url;
    MockEventSource.instances.push(this);
  }

  emit(data: object) {
    this.onmessage?.({ data: JSON.stringify(data) } as MessageEvent);
  }

  emitError() {
    this.onerror?.(new Event("error"));
  }

  close() {
    this._closed = true;
  }

  get closed() {
    return this._closed;
  }
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

function mockFetchWithJobId(jobId = "job-123") {
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ job_id: jobId }),
    })
  );
}

// ── AC01: Enter submits, Shift+Enter inserts newline ─────────────────────────

describe("US-003 AC01 — keyboard behaviour", () => {
  it("pressing Enter submits the form (calls fetch)", async () => {
    mockFetchWithJobId();
    const user = userEvent.setup();
    render(<App />);

    const textarea = screen.getByTestId("prompt-input");
    await user.type(textarea, "a landscape");
    await user.keyboard("{Enter}");

    expect(globalThis.fetch).toHaveBeenCalledTimes(1);
  });

  it("pressing Shift+Enter inserts a newline (does NOT submit)", async () => {
    mockFetchWithJobId();
    const user = userEvent.setup();
    render(<App />);

    const textarea = screen.getByTestId("prompt-input") as HTMLTextAreaElement;
    await user.type(textarea, "line one");
    await user.keyboard("{Shift>}{Enter}{/Shift}");

    // fetch not called
    expect(globalThis.fetch).not.toHaveBeenCalled();
    // value has a newline
    expect(textarea.value).toContain("\n");
  });
});

// ── AC02: User bubble appears in chat timeline ────────────────────────────────

describe("US-003 AC02 — user bubble", () => {
  it("shows a user bubble with the prompt text after submit", async () => {
    mockFetchWithJobId();
    const user = userEvent.setup();
    render(<App />);

    const textarea = screen.getByTestId("prompt-input");
    await user.type(textarea, "a sunset over mountains");
    await user.click(screen.getByRole("button", { name: "Generate" }));

    const userBubble = await screen.findByTestId("bubble-user");
    expect(userBubble).toHaveTextContent("a sunset over mountains");
  });

  it("prompt input is cleared after submit", async () => {
    mockFetchWithJobId();
    const user = userEvent.setup();
    render(<App />);

    const textarea = screen.getByTestId(
      "prompt-input"
    ) as HTMLTextAreaElement;
    await user.type(textarea, "some text");
    await user.click(screen.getByRole("button", { name: "Generate" }));

    expect(textarea.value).toBe("");
  });
});

// ── AC03: Assistant bubble with progress indicator appears immediately ────────

describe("US-003 AC03 — assistant bubble with progress indicator", () => {
  it("shows an assistant bubble immediately after submit", async () => {
    mockFetchWithJobId();
    const user = userEvent.setup();
    render(<App />);

    await user.type(screen.getByTestId("prompt-input"), "test prompt");
    await user.click(screen.getByRole("button", { name: "Generate" }));

    expect(await screen.findByTestId("bubble-assistant")).toBeInTheDocument();
  });

  it("assistant bubble shows a progress row while streaming", async () => {
    mockFetchWithJobId();
    const user = userEvent.setup();
    render(<App />);

    await user.type(screen.getByTestId("prompt-input"), "test prompt");
    await user.click(screen.getByRole("button", { name: "Generate" }));

    expect(await screen.findByTestId("progress-row")).toBeInTheDocument();
  });
});

// ── AC04: SSE progress events update assistant bubble ────────────────────────

describe("US-003 AC04 — SSE progress events", () => {
  it("updates bubble content and progress bar on each SSE event", async () => {
    mockFetchWithJobId("job-ac04");
    const user = userEvent.setup();
    render(<App />);

    await user.type(screen.getByTestId("prompt-input"), "mountains");
    await user.click(screen.getByRole("button", { name: "Generate" }));

    // Wait for the EventSource to be created (after fetch resolves)
    await waitFor(() => expect(MockEventSource.instances.length).toBeGreaterThan(0));
    const es = MockEventSource.instances[MockEventSource.instances.length - 1];

    // Verify SSE URL includes job id
    expect(es.url).toContain("job-ac04");

    act(() => {
      es.emit({ step: "sampling", pct: 0.5 });
    });

    await waitFor(() => {
      expect(screen.getByTestId("bubble-assistant")).toHaveTextContent("sampling");
    });

    const label = await screen.findByTestId("progress-label");
    expect(label).toHaveTextContent("50%");
  });

  it("marks bubble complete when step=done is received", async () => {
    mockFetchWithJobId("job-done");
    const user = userEvent.setup();
    render(<App />);

    await user.type(screen.getByTestId("prompt-input"), "final image");
    await user.click(screen.getByRole("button", { name: "Generate" }));

    await waitFor(() => expect(MockEventSource.instances.length).toBeGreaterThan(0));
    const es = MockEventSource.instances[MockEventSource.instances.length - 1];

    act(() => {
      es.emit({ step: "done", pct: 1.0 });
    });

    await waitFor(() => {
      expect(screen.getByTestId("bubble-assistant")).toHaveTextContent(
        "Generation complete"
      );
    });
    // No progress row after completion
    expect(screen.queryByTestId("progress-row")).not.toBeInTheDocument();
  });

  it("marks bubble as error when step=error is received", async () => {
    mockFetchWithJobId("job-err");
    const user = userEvent.setup();
    render(<App />);

    await user.type(screen.getByTestId("prompt-input"), "broken");
    await user.click(screen.getByRole("button", { name: "Generate" }));

    await waitFor(() => expect(MockEventSource.instances.length).toBeGreaterThan(0));
    const es = MockEventSource.instances[MockEventSource.instances.length - 1];

    act(() => {
      es.emit({ step: "error", pct: 0.0, error: "VRAM out of memory" });
    });

    await waitFor(() => {
      expect(screen.getByTestId("bubble-assistant")).toHaveTextContent(
        "VRAM out of memory"
      );
    });
    expect(screen.queryByTestId("progress-row")).not.toBeInTheDocument();
  });
});

// ── AC05: Connection lost message ─────────────────────────────────────────────

describe("US-003 AC05 — connection lost", () => {
  it("shows connection-lost message when SSE errors before completion", async () => {
    mockFetchWithJobId("job-lost");
    const user = userEvent.setup();
    render(<App />);

    await user.type(screen.getByTestId("prompt-input"), "anything");
    await user.click(screen.getByRole("button", { name: "Generate" }));

    await waitFor(() => expect(MockEventSource.instances.length).toBeGreaterThan(0));
    const es = MockEventSource.instances[MockEventSource.instances.length - 1];

    act(() => {
      es.emitError();
    });

    await waitFor(() => {
      expect(screen.getByTestId("bubble-assistant")).toHaveTextContent(
        "Connection lost — check job status."
      );
    });
  });
});

// ── AC06: Disabled while in-flight ───────────────────────────────────────────

describe("US-003 AC06 — disabled while submitting", () => {
  it("textarea and button are disabled while a job is in-flight", async () => {
    mockFetchWithJobId();
    const user = userEvent.setup();
    render(<App />);

    await user.type(screen.getByTestId("prompt-input"), "anything");
    await user.click(screen.getByRole("button", { name: "Generate" }));

    // Immediately after submit (fetch is resolving) both should be disabled
    expect(screen.getByTestId("prompt-input")).toBeDisabled();
    expect(screen.getByRole("button", { name: "Generate" })).toBeDisabled();
  });

  it("re-enables input after SSE stream completes", async () => {
    mockFetchWithJobId("job-reenable");
    const user = userEvent.setup();
    render(<App />);

    await user.type(screen.getByTestId("prompt-input"), "anything");
    await user.click(screen.getByRole("button", { name: "Generate" }));

    await waitFor(() => expect(MockEventSource.instances.length).toBeGreaterThan(0));
    const es = MockEventSource.instances[MockEventSource.instances.length - 1];

    act(() => {
      es.emit({ step: "done", pct: 1.0 });
    });

    await waitFor(() => {
      expect(screen.getByTestId("prompt-input")).not.toBeDisabled();
      expect(screen.getByRole("button", { name: "Generate" })).not.toBeDisabled();
    });
  });

  it("clicking Generate while in-flight does not submit again", async () => {
    mockFetchWithJobId();
    const user = userEvent.setup();
    render(<App />);

    await user.type(screen.getByTestId("prompt-input"), "anything");
    await user.click(screen.getByRole("button", { name: "Generate" }));

    // Button is disabled; further clicks should not call fetch again
    await user.click(screen.getByRole("button", { name: "Generate" }));
    expect(globalThis.fetch).toHaveBeenCalledTimes(1);
  });
});
