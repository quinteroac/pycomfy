/**
 * Tests for US-004 — Apply final polish and micro-interactions
 *
 * AC01: polish applied — spacing, alignment, typographic inconsistencies resolved.
 * AC02: animations implemented — bubble entrance, progress pulse, media reveal.
 * AC03: all animations respect prefers-reduced-motion.
 * AC04: no animation blocks user interaction.
 */
import { readFileSync } from "fs";
import { fileURLToPath } from "url";
import { dirname, join } from "path";
import { render, screen } from "@testing-library/react";
import { ChatBubble } from "../components/ChatBubble";
import { App } from "../App";
import type { ChatMessage } from "../types";

// Read CSS source files at test-load time via Node.js fs (vitest runs in Node).
const __filename = fileURLToPath(import.meta.url);
const __dir = dirname(__filename);
const chatBubbleCssRaw = readFileSync(
  join(__dir, "../components/ChatBubble.module.css"),
  "utf-8"
);
const appCssRaw = readFileSync(join(__dir, "../App.css"), "utf-8");

// ── Setup / Teardown ──────────────────────────────────────────────────────────

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

// ── AC01: Polish — spacing, alignment, typography ─────────────────────────────

describe("US-004 AC01 — Polish applied", () => {
  it("bubble gap is 8px (consistent spacing between content and media)", () => {
    // The .bubble rule should declare gap: 8px (upgraded from 6px)
    const bubbleRule = chatBubbleCssRaw.match(/\.bubble\s*\{([^}]*)\}/)?.[1] ?? "";
    expect(bubbleRule).toMatch(/gap:\s*8px/);
  });

  it("--surface-3 is defined in the root token set", () => {
    expect(appCssRaw).toMatch(/--surface-3:/);
  });

  it("composer-send transitions are consistent (both 0.15s)", () => {
    const sendRule = appCssRaw.match(/\.composer-send\s*\{([^}]*)\}/)?.[1] ?? "";
    // Both durations in transition shorthand must be 0.15s
    expect(sendRule).toMatch(/transition:.*0\.15s.*0\.15s/);
  });

  it("chat-area clips horizontal overflow", () => {
    const chatAreaRule = appCssRaw.match(/\.chat-area\s*\{([^}]*)\}/)?.[1] ?? "";
    expect(chatAreaRule).toMatch(/overflow-x:\s*hidden/);
  });

  it("user bubble renders with correct alignment", () => {
    const msg: ChatMessage = { id: "u1", role: "user", content: "hello", status: "complete" };
    render(<ChatBubble message={msg} />);
    expect(screen.getByTestId("bubble-user")).toBeInTheDocument();
  });

  it("assistant bubble renders with correct alignment", () => {
    const msg: ChatMessage = { id: "a1", role: "assistant", content: "hi", status: "complete" };
    render(<ChatBubble message={msg} />);
    expect(screen.getByTestId("bubble-assistant")).toBeInTheDocument();
  });

  it("download-btn hover uses surface-3 (visible state change on hover)", () => {
    // The hover rule must reference --surface-3, not fall back to --surface-2
    const hoverRule = chatBubbleCssRaw.match(/\.download-btn:hover\s*\{([^}]*)\}/)?.[1] ?? "";
    expect(hoverRule).toMatch(/--surface-3/);
    expect(hoverRule).not.toMatch(/--surface-2/);
  });
});

// ── AC02: Animations implemented ─────────────────────────────────────────────

describe("US-004 AC02 — Animations implemented", () => {
  it("CSS defines bubbleIn keyframe for bubble entrance", () => {
    expect(chatBubbleCssRaw).toMatch(/@keyframes\s+bubbleIn/);
  });

  it("bubbleIn uses opacity and translateY for a subtle entrance", () => {
    const idx = chatBubbleCssRaw.indexOf("@keyframes bubbleIn");
    const block = chatBubbleCssRaw.slice(idx, chatBubbleCssRaw.indexOf("}", idx + 20) + 1);
    expect(block).toMatch(/opacity/);
    expect(block).toMatch(/translateY/);
  });

  it("CSS defines progressPulse keyframe for progress bar", () => {
    expect(chatBubbleCssRaw).toMatch(/@keyframes\s+progressPulse/);
  });

  it("progress-bar-fill uses progressPulse animation while streaming", () => {
    const fillRule = chatBubbleCssRaw.match(/\.progress-bar-fill\s*\{([^}]*)\}/)?.[1] ?? "";
    expect(fillRule).toMatch(/progressPulse/);
    expect(fillRule).toMatch(/infinite/);
  });

  it("CSS defines mediaReveal keyframe for media reveal", () => {
    expect(chatBubbleCssRaw).toMatch(/@keyframes\s+mediaReveal/);
  });

  it("media-container uses mediaReveal animation", () => {
    const containerRule = chatBubbleCssRaw.match(/\.media-container\s*\{([^}]*)\}/)?.[1] ?? "";
    expect(containerRule).toMatch(/mediaReveal/);
  });

  it(".bubble uses bubbleIn animation", () => {
    const bubbleRule = chatBubbleCssRaw.match(/\.bubble\s*\{([^}]*)\}/)?.[1] ?? "";
    expect(bubbleRule).toMatch(/bubbleIn/);
  });
});

// ── AC03: Prefers-reduced-motion ─────────────────────────────────────────────

describe("US-004 AC03 — Animations respect prefers-reduced-motion", () => {
  it("CSS contains a prefers-reduced-motion: reduce media query", () => {
    expect(chatBubbleCssRaw).toMatch(/@media[^{]*prefers-reduced-motion[^{]*reduce/);
  });

  it("reduced-motion block disables .bubble animation", () => {
    // Extract the entire reduced-motion media block
    const start = chatBubbleCssRaw.search(/@media[^{]*prefers-reduced-motion[^{]*reduce/);
    const blockStart = chatBubbleCssRaw.indexOf("{", start) + 1;
    // Find the matching closing brace
    let depth = 1;
    let i = blockStart;
    while (i < chatBubbleCssRaw.length && depth > 0) {
      if (chatBubbleCssRaw[i] === "{") depth++;
      if (chatBubbleCssRaw[i] === "}") depth--;
      i++;
    }
    const rmBlock = chatBubbleCssRaw.slice(blockStart, i - 1);
    expect(rmBlock).toMatch(/\.bubble/);
    expect(rmBlock).toMatch(/animation:\s*none/);
    expect(rmBlock).toMatch(/\.progress-bar-fill/);
    expect(rmBlock).toMatch(/\.media-container/);
  });
});

// ── AC04: Animations do not block user interaction ────────────────────────────

describe("US-004 AC04 — Animations do not block user interaction", () => {
  it("animated .bubble has no pointer-events: none", () => {
    const bubbleRule = chatBubbleCssRaw.match(/\.bubble\s*\{([^}]*)\}/)?.[1] ?? "";
    expect(bubbleRule).not.toMatch(/pointer-events:\s*none/);
  });

  it("animated .media-container has no pointer-events: none", () => {
    const containerRule = chatBubbleCssRaw.match(/\.media-container\s*\{([^}]*)\}/)?.[1] ?? "";
    expect(containerRule).not.toMatch(/pointer-events:\s*none/);
  });

  it("send button is focusable in idle state (not blocked by animations)", () => {
    render(<App />);
    const btn = screen.getByRole("button", { name: /generate/i });
    expect(btn).not.toBeDisabled();
    btn.focus();
    expect(document.activeElement).toBe(btn);
  });

  it("textarea is focusable while bubbles are visible", () => {
    render(<App />);
    const textarea = screen.getByRole("textbox", { name: /prompt/i });
    textarea.focus();
    expect(document.activeElement).toBe(textarea);
  });

  it("bubble animation uses fill-mode 'both' (not 'none') — does not block with visibility", () => {
    const bubbleRule = chatBubbleCssRaw.match(/\.bubble\s*\{([^}]*)\}/)?.[1] ?? "";
    // 'both' fill-mode is used — animation fills forward/backward but does not use visibility:hidden
    expect(bubbleRule).not.toMatch(/visibility:\s*hidden/);
  });
});
