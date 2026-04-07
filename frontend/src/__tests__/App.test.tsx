import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { App } from "../App";

// Mock URL APIs
beforeEach(() => {
  Object.defineProperty(globalThis, "URL", {
    value: {
      createObjectURL: vi.fn(() => "blob:mock-url"),
      revokeObjectURL: vi.fn(),
    },
    writable: true,
    configurable: true,
  });
  // Mock fetch
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: true }));
});

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

async function selectPipeline(user: ReturnType<typeof userEvent.setup>, pipeline: string) {
  // First switch to video
  const videoBtn = screen.getByTestId("media-type-video");
  await user.click(videoBtn);
  // Then select pipeline
  const sel = screen.getByTestId("select-pipeline") as HTMLSelectElement;
  await user.selectOptions(sel, pipeline);
}

describe("App — US-002 Image Upload integration", () => {
  // ── AC01: upload control appears only for i2v, is2v, ia2v ─────────────────

  it("AC01: image upload control is NOT shown for default image media type", () => {
    render(<App />);
    expect(screen.queryByTestId("image-upload")).not.toBeInTheDocument();
  });

  it("AC01: image upload control is NOT shown for t2v pipeline", async () => {
    const user = userEvent.setup();
    render(<App />);
    await selectPipeline(user, "t2v");
    expect(screen.queryByTestId("image-upload")).not.toBeInTheDocument();
  });

  it("AC01: image upload control is NOT shown for flf2v pipeline", async () => {
    const user = userEvent.setup();
    render(<App />);
    await selectPipeline(user, "flf2v");
    expect(screen.queryByTestId("image-upload")).not.toBeInTheDocument();
  });

  it("AC01: image upload control IS shown for i2v pipeline", async () => {
    const user = userEvent.setup();
    render(<App />);
    await selectPipeline(user, "i2v");
    expect(screen.getByTestId("image-upload")).toBeInTheDocument();
  });

  it("AC01: image upload control IS shown for is2v pipeline", async () => {
    const user = userEvent.setup();
    render(<App />);
    await selectPipeline(user, "is2v");
    expect(screen.getByTestId("image-upload")).toBeInTheDocument();
  });

  it("AC01: image upload control IS shown for ia2v pipeline", async () => {
    const user = userEvent.setup();
    render(<App />);
    await selectPipeline(user, "ia2v");
    expect(screen.getByTestId("image-upload")).toBeInTheDocument();
  });

  // ── AC01: upload control disappears when switching away ────────────────────

  it("AC01: upload control disappears when switching back to t2v", async () => {
    const user = userEvent.setup();
    render(<App />);
    await selectPipeline(user, "i2v");
    expect(screen.getByTestId("image-upload")).toBeInTheDocument();
    await user.selectOptions(screen.getByTestId("select-pipeline"), "t2v");
    expect(screen.queryByTestId("image-upload")).not.toBeInTheDocument();
  });

  it("AC01: upload control disappears when switching to audio media type", async () => {
    const user = userEvent.setup();
    render(<App />);
    await selectPipeline(user, "i2v");
    await user.click(screen.getByTestId("media-type-audio"));
    expect(screen.queryByTestId("image-upload")).not.toBeInTheDocument();
  });

  // ── AC04: sends multipart/form-data ───────────────────────────────────────

  it("AC04: sends FormData with image when file is selected and i2v is active", async () => {
    const user = userEvent.setup();
    render(<App />);
    await selectPipeline(user, "i2v");

    const input = screen.getByTestId("image-upload-input");
    const file = new File(["img"], "frame.jpg", { type: "image/jpeg" });
    await user.upload(input, file);

    await user.click(screen.getByRole("button", { name: "Generate" }));

    expect(globalThis.fetch).toHaveBeenCalledTimes(1);
    const [, options] = (globalThis.fetch as ReturnType<typeof vi.fn>).mock.calls[0];
    expect(options.body).toBeInstanceOf(FormData);
    const fd = options.body as FormData;
    expect(fd.get("image")).toBe(file);
    expect(fd.get("pipeline")).toBe("i2v");
  });

  // ── AC05: validation — no file + requires image → error shown ─────────────

  it("AC05: shows inline error when submitting without image on i2v", async () => {
    const user = userEvent.setup();
    render(<App />);
    await selectPipeline(user, "i2v");

    await user.click(screen.getByRole("button", { name: "Generate" }));

    expect(screen.getByTestId("image-upload-error")).toBeInTheDocument();
    expect(globalThis.fetch).not.toHaveBeenCalled();
  });

  it("AC05: shows inline error for ia2v when no image selected", async () => {
    const user = userEvent.setup();
    render(<App />);
    await selectPipeline(user, "ia2v");

    await user.click(screen.getByRole("button", { name: "Generate" }));

    expect(screen.getByTestId("image-upload-error")).toBeInTheDocument();
    expect(globalThis.fetch).not.toHaveBeenCalled();
  });

  it("AC05: error is cleared and fetch is called after a file is added", async () => {
    const user = userEvent.setup();
    render(<App />);
    await selectPipeline(user, "i2v");

    // First submit without file → error
    await user.click(screen.getByRole("button", { name: "Generate" }));
    expect(screen.getByTestId("image-upload-error")).toBeInTheDocument();

    // Now upload a file and resubmit
    const input = screen.getByTestId("image-upload-input");
    await user.upload(input, new File(["x"], "x.png", { type: "image/png" }));
    await user.click(screen.getByRole("button", { name: "Generate" }));

    expect(screen.queryByTestId("image-upload-error")).not.toBeInTheDocument();
    expect(globalThis.fetch).toHaveBeenCalledTimes(1);
  });
});
