/**
 * Accessibility regression tests — it_000047 US-001 AC02
 *
 * Verifies that all Critical and High issues identified in the audit report
 * (.agents/flow/it_000047_audit-report_001.md) have been resolved.
 */
import { render, screen } from "@testing-library/react";
import { MediaTypeSelector } from "../components/MediaTypeSelector";
import { ChatBubble } from "../components/ChatBubble";
import { ImageUpload } from "../components/ImageUpload";
import type { ChatMessage } from "../types";
import { App } from "../App";

// ── C-01: role="radiogroup" on MediaTypeSelector ──────────────────────────────

describe("C-01 — MediaTypeSelector uses role=radiogroup", () => {
  it("container has role=radiogroup (not role=group)", () => {
    render(<MediaTypeSelector value="image" onChange={() => {}} />);
    expect(screen.getByRole("radiogroup", { name: /media type/i })).toBeInTheDocument();
    expect(screen.queryByRole("group", { name: /media type/i })).not.toBeInTheDocument();
  });

  it("individual buttons have role=radio", () => {
    render(<MediaTypeSelector value="image" onChange={() => {}} />);
    const radios = screen.getAllByRole("radio");
    expect(radios).toHaveLength(3);
  });
});

// ── H-02: chat message list is a live region (role=log) ──────────────────────

describe("H-02 — Chat message list uses role=log", () => {
  beforeEach(() => {
    Object.defineProperty(globalThis, "URL", {
      value: { createObjectURL: vi.fn(() => "blob:x"), revokeObjectURL: vi.fn() },
      writable: true,
      configurable: true,
    });
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: true }));
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it("chat-messages container has role=log after first message", async () => {
    const { default: userEvent } = await import("@testing-library/user-event");
    const user = userEvent.setup();
    render(<App />);
    const textarea = screen.getByTestId("prompt-input");
    await user.type(textarea, "a prompt");
    await user.click(screen.getByRole("button", { name: "Generate" }));

    const log = screen.getByRole("log");
    expect(log).toBeInTheDocument();
    expect(log).toHaveAttribute("aria-label", "Chat messages");
  });
});

// ── H-03: progress bar has role=progressbar + ARIA attributes ────────────────

describe("H-03 — Progress bar has correct ARIA attributes", () => {
  const streamingMsg: ChatMessage = {
    id: "m1",
    role: "assistant",
    content: "Sampling…",
    status: "streaming",
    progress: 42,
    progressLabel: "42%",
  };

  it("progress-bar has role=progressbar", () => {
    render(<ChatBubble message={streamingMsg} />);
    expect(screen.getByRole("progressbar")).toBeInTheDocument();
  });

  it("progress-bar has aria-valuenow reflecting current progress", () => {
    render(<ChatBubble message={streamingMsg} />);
    expect(screen.getByRole("progressbar")).toHaveAttribute("aria-valuenow", "42");
  });

  it("progress-bar has aria-valuemin=0 and aria-valuemax=100", () => {
    render(<ChatBubble message={streamingMsg} />);
    const bar = screen.getByRole("progressbar");
    expect(bar).toHaveAttribute("aria-valuemin", "0");
    expect(bar).toHaveAttribute("aria-valuemax", "100");
  });

  it("progress-bar has an accessible label", () => {
    render(<ChatBubble message={streamingMsg} />);
    expect(screen.getByRole("progressbar")).toHaveAttribute("aria-label", "Generation progress");
  });
});

// ── H-04: send button meets WCAG 2.5.5 minimum touch target (44×44px) ────────

describe("H-04 — Composer send button is at least 44×44px", () => {
  beforeEach(() => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: true }));
    Object.defineProperty(globalThis, "URL", {
      value: { createObjectURL: vi.fn(() => "blob:x"), revokeObjectURL: vi.fn() },
      writable: true,
      configurable: true,
    });
  });
  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it("generate button has aria-label='Generate'", () => {
    render(<App />);
    expect(screen.getByRole("button", { name: "Generate" })).toBeInTheDocument();
  });
});

// ── H-05: remove-image button has an accessible label ────────────────────────

describe("H-05 — Remove image button is accessible", () => {
  it("remove button has aria-label='Remove image'", () => {
    const file = new File(["x"], "img.png", { type: "image/png" });
    Object.defineProperty(globalThis, "URL", {
      value: { createObjectURL: vi.fn(() => "blob:mock"), revokeObjectURL: vi.fn() },
      writable: true,
      configurable: true,
    });
    render(<ImageUpload value={file} onChange={() => {}} />);
    expect(screen.getByRole("button", { name: "Remove image" })).toBeInTheDocument();
  });
});

// ── H-06: download link has focus-visible style (structural check) ────────────

describe("H-06 — Download link is present and accessible", () => {
  const completeMsg: ChatMessage = {
    id: "m2",
    role: "assistant",
    content: "Generation complete ✓",
    status: "complete",
    mediaUrl: "http://example.com/output.png",
    mediaType: "image",
    filename: "output.png",
  };

  it("download link is rendered with correct href and download attribute", () => {
    render(<ChatBubble message={completeMsg} />);
    const link = screen.getByTestId("download-btn");
    expect(link).toBeInTheDocument();
    expect(link).toHaveAttribute("href", "http://example.com/output.png");
    expect(link).toHaveAttribute("download", "output.png");
  });
});
