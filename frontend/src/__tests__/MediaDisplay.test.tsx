/**
 * Tests for US-004 — Display generated media inline and allow download
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

function mockFetchWithJobId(jobId = "job-media") {
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ job_id: jobId }),
    })
  );
}

async function submitAndGetES(jobId = "job-media") {
  mockFetchWithJobId(jobId);
  const user = userEvent.setup();
  render(<App />);
  await user.type(screen.getByTestId("prompt-input"), "test");
  await user.click(screen.getByRole("button", { name: "Generate" }));
  await waitFor(() => expect(MockEventSource.instances.length).toBeGreaterThan(0));
  return MockEventSource.instances[MockEventSource.instances.length - 1];
}

// ── AC01: Generated media rendered inline ────────────────────────────────────

describe("US-004 AC01 — media rendered inline on completion", () => {
  it("shows an <img> element when image job completes", async () => {
    const es = await submitAndGetES("job-ac01-img");

    act(() => {
      es.emit({ step: "done", pct: 1.0 });
    });

    await waitFor(() => {
      expect(screen.getByTestId("media-image")).toBeInTheDocument();
    });
  });

  it("image src contains the job ID", async () => {
    const es = await submitAndGetES("job-src-check");

    act(() => {
      es.emit({ step: "done", pct: 1.0 });
    });

    await waitFor(() => {
      expect(screen.getByTestId("media-image")).toHaveAttribute(
        "src",
        expect.stringContaining("job-src-check")
      );
    });
  });

  it("progress row is gone after completion", async () => {
    const es = await submitAndGetES("job-ac01-pr");

    act(() => {
      es.emit({ step: "done", pct: 1.0 });
    });

    await waitFor(() => {
      expect(screen.queryByTestId("progress-row")).not.toBeInTheDocument();
    });
  });

  it("media container appears inside the assistant bubble", async () => {
    const es = await submitAndGetES("job-ac01-container");

    act(() => {
      es.emit({ step: "done", pct: 1.0 });
    });

    await waitFor(() => {
      expect(screen.getByTestId("media-container")).toBeInTheDocument();
    });

    const bubble = screen.getByTestId("bubble-assistant");
    expect(bubble).toContainElement(screen.getByTestId("media-container"));
  });
});

// ── AC02: Download button ─────────────────────────────────────────────────────

describe("US-004 AC02 — download button", () => {
  it("shows a Download button after job completes", async () => {
    const es = await submitAndGetES("job-ac02-btn");

    act(() => {
      es.emit({ step: "done", pct: 1.0 });
    });

    await waitFor(() => {
      expect(screen.getByTestId("download-btn")).toBeInTheDocument();
    });
    expect(screen.getByTestId("download-btn")).toHaveTextContent("Download");
  });

  it("download button has a .png extension for image jobs", async () => {
    const es = await submitAndGetES("job-ac02-ext");

    act(() => {
      es.emit({ step: "done", pct: 1.0 });
    });

    await waitFor(() => {
      const btn = screen.getByTestId("download-btn");
      expect(btn).toHaveAttribute("download", expect.stringMatching(/\.png$/));
    });
  });

  it("download button href points to the result endpoint containing job ID", async () => {
    const es = await submitAndGetES("job-ac02-href");

    act(() => {
      es.emit({ step: "done", pct: 1.0 });
    });

    await waitFor(() => {
      const btn = screen.getByTestId("download-btn");
      expect(btn).toHaveAttribute("href", expect.stringContaining("job-ac02-href"));
      expect(btn).toHaveAttribute("href", expect.stringContaining("/result"));
    });
  });

  it("download button is not shown while streaming", async () => {
    await submitAndGetES("job-ac02-no-dl");

    expect(screen.queryByTestId("download-btn")).not.toBeInTheDocument();
  });
});

// ── AC03: Error message shown on failure ──────────────────────────────────────

describe("US-004 AC03 — error message on failure", () => {
  it("shows the error message from the server when step=error", async () => {
    const es = await submitAndGetES("job-ac03-err");

    act(() => {
      es.emit({ step: "error", pct: 0.0, error: "VRAM out of memory" });
    });

    await waitFor(() => {
      expect(screen.getByTestId("bubble-assistant")).toHaveTextContent(
        "VRAM out of memory"
      );
    });
  });

  it("no media element is shown when job fails", async () => {
    const es = await submitAndGetES("job-ac03-no-media");

    act(() => {
      es.emit({ step: "error", pct: 0.0, error: "pipeline error" });
    });

    await waitFor(() => {
      expect(screen.getByTestId("bubble-assistant")).toHaveTextContent("pipeline error");
    });

    expect(screen.queryByTestId("media-image")).not.toBeInTheDocument();
    expect(screen.queryByTestId("media-video")).not.toBeInTheDocument();
    expect(screen.queryByTestId("media-audio")).not.toBeInTheDocument();
    expect(screen.queryByTestId("download-btn")).not.toBeInTheDocument();
  });

  it("falls back to generic error text when server error field is null", async () => {
    const es = await submitAndGetES("job-ac03-generic");

    act(() => {
      es.emit({ step: "error", pct: 0.0, error: null });
    });

    await waitFor(() => {
      expect(screen.getByTestId("bubble-assistant")).toHaveTextContent(
        "Generation failed."
      );
    });
  });
});

// ── AC04: Chat history persists in-memory ────────────────────────────────────

describe("US-004 AC04 — in-memory chat history", () => {
  it("retains all previous messages after a second submission", async () => {
    mockFetchWithJobId("job-first");
    const user = userEvent.setup();
    render(<App />);

    // First submission
    await user.type(screen.getByTestId("prompt-input"), "first prompt");
    await user.click(screen.getByRole("button", { name: "Generate" }));

    await waitFor(() => expect(MockEventSource.instances.length).toBe(1));
    act(() => {
      MockEventSource.instances[0].emit({ step: "done", pct: 1.0 });
    });

    // Wait for input to re-enable before second submission
    await waitFor(() =>
      expect(screen.getByTestId("prompt-input")).not.toBeDisabled()
    );

    // Second submission
    mockFetchWithJobId("job-second");
    await user.type(screen.getByTestId("prompt-input"), "second prompt");
    await user.click(screen.getByRole("button", { name: "Generate" }));

    const userBubbles = screen.getAllByTestId("bubble-user");
    expect(userBubbles.length).toBe(2);
    expect(userBubbles[0]).toHaveTextContent("first prompt");
    expect(userBubbles[1]).toHaveTextContent("second prompt");
  });

  it("does not write to localStorage", async () => {
    const setSpy = vi.spyOn(Storage.prototype, "setItem");
    const es = await submitAndGetES("job-ls");

    act(() => {
      es.emit({ step: "done", pct: 1.0 });
    });

    await waitFor(() =>
      expect(screen.getByTestId("prompt-input")).not.toBeDisabled()
    );

    expect(setSpy).not.toHaveBeenCalled();
  });
});
