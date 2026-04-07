/**
 * Distill US-002 — "Distill the interface to its essential structure"
 *
 * AC01: distill recommendations are applied; accepted changes are implemented.
 * AC02: chat input, media type selector, and parameter panel are visually
 *       distinct without competing for attention.
 * AC03: no decorative elements remain that do not serve a functional or
 *       communicative purpose.
 */
import { render, screen } from "@testing-library/react";
import { App } from "../App";
import { MediaTypeSelector } from "../components/MediaTypeSelector";
import { ParameterPanel } from "../components/ParameterPanel";
import { useGenerationParams } from "../hooks/useGenerationParams";
import { renderHook } from "@testing-library/react";

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

// ── AC01: distillation applied ────────────────────────────────────────────────

describe("US-002 AC01 — Distillation applied", () => {
  it("renders without errors after distillation", () => {
    expect(() => render(<App />)).not.toThrow();
  });

  it("empty-state hint is concise (≤ 60 characters)", () => {
    render(<App />);
    const hint = screen.getByText(/type a prompt/i);
    expect(hint.textContent!.length).toBeLessThanOrEqual(60);
  });
});

// ── AC02: three sections are visually distinct without competing ──────────────

describe("US-002 AC02 — Sections distinct without competing", () => {
  it("MediaTypeSelector renders as a labelled radiogroup", () => {
    render(<MediaTypeSelector value="image" onChange={() => {}} />);
    expect(screen.getByRole("radiogroup", { name: /media type/i })).toBeInTheDocument();
  });

  it("MediaTypeSelector has three distinct radio options", () => {
    render(<MediaTypeSelector value="image" onChange={() => {}} />);
    expect(screen.getAllByRole("radio")).toHaveLength(3);
  });

  it("ParameterPanel renders with labelled controls", () => {
    const { result } = renderHook(() => useGenerationParams());
    render(<ParameterPanel params={result.current.params} onUpdate={result.current.updateParam} />);
    expect(screen.getByTestId("parameter-panel")).toBeInTheDocument();
    expect(screen.getByRole("combobox", { name: /model/i })).toBeInTheDocument();
  });

  it("prompt textarea is present and labelled", () => {
    render(<App />);
    expect(screen.getByRole("textbox", { name: /prompt/i })).toBeInTheDocument();
  });

  it("generate button is the sole primary call-to-action", () => {
    render(<App />);
    const primaryButtons = screen.getAllByRole("button", { name: /generate/i });
    expect(primaryButtons).toHaveLength(1);
  });
});

// ── AC03: no decorative elements without functional purpose ───────────────────

describe("US-002 AC03 — No purely decorative elements", () => {
  it("decorative logo glyph ◈ is not rendered", () => {
    render(<App />);
    expect(screen.queryByText("◈")).not.toBeInTheDocument();
  });

  it("decorative divider element is not rendered", () => {
    render(<App />);
    expect(document.querySelector(".composer-divider")).not.toBeInTheDocument();
  });

  it("app header still renders the Parallax title", () => {
    render(<App />);
    expect(screen.getByText("Parallax")).toBeInTheDocument();
  });
});
