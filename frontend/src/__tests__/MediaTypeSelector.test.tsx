
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MediaTypeSelector } from "../components/MediaTypeSelector";
import { MediaType } from "../types";

function setup(value: MediaType = "image", onChange: (t: MediaType) => void = () => {}) {
  return render(<MediaTypeSelector value={value} onChange={onChange} />);
}

describe("MediaTypeSelector — US-001", () => {
  it("AC01: renders image, video, and audio buttons", () => {
    setup();
    expect(screen.getByTestId("media-type-image")).toBeInTheDocument();
    expect(screen.getByTestId("media-type-video")).toBeInTheDocument();
    expect(screen.getByTestId("media-type-audio")).toBeInTheDocument();
  });

  it("AC01: the group is labelled for accessibility", () => {
    setup();
    expect(screen.getByRole("radiogroup", { name: /media type/i })).toBeInTheDocument();
  });

  it("AC01: selected type has aria-checked=true, others false", () => {
    setup("video");
    expect(screen.getByTestId("media-type-video")).toHaveAttribute(
      "aria-checked",
      "true"
    );
    expect(screen.getByTestId("media-type-image")).toHaveAttribute(
      "aria-checked",
      "false"
    );
    expect(screen.getByTestId("media-type-audio")).toHaveAttribute(
      "aria-checked",
      "false"
    );
  });

  it("AC01: clicking a type calls onChange with the correct value", async () => {
    const user = userEvent.setup();
    let received: MediaType | null = null;
    setup("image", (t) => {
      received = t;
    });
    await user.click(screen.getByTestId("media-type-audio"));
    expect(received).toBe("audio");
  });

  it("AC01: all three labels are visible", () => {
    setup();
    expect(screen.getByText("Image")).toBeInTheDocument();
    expect(screen.getByText("Video")).toBeInTheDocument();
    expect(screen.getByText("Audio")).toBeInTheDocument();
  });
});
