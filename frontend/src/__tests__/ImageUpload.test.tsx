import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ImageUpload } from "../components/ImageUpload";
import { useState } from "react";

// Mock URL APIs unavailable in happy-dom
beforeEach(() => {
  Object.defineProperty(globalThis, "URL", {
    value: {
      createObjectURL: vi.fn(() => "blob:mock-preview-url"),
      revokeObjectURL: vi.fn(),
    },
    writable: true,
    configurable: true,
  });
});

function Harness({
  initial = null,
  error,
}: {
  initial?: File | null;
  error?: string;
}) {
  const [value, setValue] = useState<File | null>(initial);
  return <ImageUpload value={value} onChange={setValue} error={error} />;
}

function makeImageFile(name = "photo.jpg", type = "image/jpeg"): File {
  return new File(["data"], name, { type });
}

describe("ImageUpload — US-002", () => {
  // ── AC01: upload control rendered ─────────────────────────────────────────

  it("AC01: renders the upload trigger button", () => {
    render(<Harness />);
    expect(screen.getByTestId("image-upload-trigger")).toBeInTheDocument();
  });

  it("AC01: renders the hidden file input", () => {
    render(<Harness />);
    expect(screen.getByTestId("image-upload-input")).toBeInTheDocument();
  });

  // ── AC02: accepted file types ──────────────────────────────────────────────

  it("AC02: file input accepts .jpg,.jpeg,.png,.webp", () => {
    render(<Harness />);
    const input = screen.getByTestId("image-upload-input") as HTMLInputElement;
    expect(input.accept).toBe(".jpg,.jpeg,.png,.webp");
  });

  // ── AC03: thumbnail preview after selecting a file ─────────────────────────

  it("AC03: shows thumbnail after a file is selected", async () => {
    const user = userEvent.setup();
    render(<Harness />);
    const input = screen.getByTestId("image-upload-input");
    const file = makeImageFile("sunset.jpg");
    await user.upload(input, file);
    expect(screen.getByTestId("image-thumbnail")).toBeInTheDocument();
  });

  it("AC03: hides the upload trigger after a file is selected", async () => {
    const user = userEvent.setup();
    render(<Harness />);
    const input = screen.getByTestId("image-upload-input");
    await user.upload(input, makeImageFile());
    expect(screen.queryByTestId("image-upload-trigger")).not.toBeInTheDocument();
  });

  it("AC03: thumbnail uses createObjectURL-generated src", async () => {
    const user = userEvent.setup();
    render(<Harness />);
    const input = screen.getByTestId("image-upload-input");
    await user.upload(input, makeImageFile("cat.png", "image/png"));
    const img = screen.getByTestId("image-thumbnail") as HTMLImageElement;
    expect(img.src).toContain("mock-preview-url");
  });

  it("AC03: remove button clears the thumbnail", async () => {
    const user = userEvent.setup();
    render(<Harness />);
    const input = screen.getByTestId("image-upload-input");
    await user.upload(input, makeImageFile());
    await user.click(screen.getByTestId("image-upload-remove"));
    expect(screen.queryByTestId("image-thumbnail")).not.toBeInTheDocument();
    expect(screen.getByTestId("image-upload-trigger")).toBeInTheDocument();
  });

  // ── AC05: inline error message ─────────────────────────────────────────────

  it("AC05: shows error message when error prop is provided", () => {
    render(<Harness error="Please select a reference image before generating." />);
    expect(screen.getByTestId("image-upload-error")).toBeInTheDocument();
    expect(screen.getByTestId("image-upload-error")).toHaveTextContent(
      "Please select a reference image before generating."
    );
  });

  it("AC05: does not show error message when error is null", () => {
    render(<Harness error={undefined} />);
    expect(screen.queryByTestId("image-upload-error")).not.toBeInTheDocument();
  });

  it("AC05: error has role=alert for accessibility", () => {
    render(<Harness error="Required." />);
    expect(screen.getByRole("alert")).toBeInTheDocument();
  });
});
