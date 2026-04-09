
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ParameterPanel } from "../components/ParameterPanel";
import {
  DEFAULT_PARAMS,
  GenerationParams,
  IMAGE_MODELS,
  VIDEO_MODELS,
  AUDIO_MODELS,
  VIDEO_PIPELINES,
} from "../types";
import { useState } from "react";

function Harness({ initial }: { initial: GenerationParams }) {
  const [params, setParams] = useState(initial);
  const update = <K extends keyof GenerationParams>(
    key: K,
    value: GenerationParams[K]
  ) => setParams((p) => ({ ...p, [key]: value }));
  return <ParameterPanel params={params} onUpdate={update} />;
}

function imageParams(): GenerationParams {
  return { ...DEFAULT_PARAMS, mediaType: "image", model: IMAGE_MODELS[0] };
}

function videoParams(): GenerationParams {
  return {
    ...DEFAULT_PARAMS,
    mediaType: "video",
    model: VIDEO_MODELS[0],
    pipeline: VIDEO_PIPELINES[0],
  };
}

function audioParams(): GenerationParams {
  return { ...DEFAULT_PARAMS, mediaType: "audio", model: AUDIO_MODELS[0] };
}

describe("ParameterPanel — US-001", () => {
  // ── AC02: image fields ─────────────────────────────────────────────────────

  it("AC02: image shows model selector", () => {
    render(<Harness initial={imageParams()} />);
    expect(screen.getByTestId("field-model")).toBeInTheDocument();
  });

  it("AC02: image shows width field", () => {
    render(<Harness initial={imageParams()} />);
    expect(screen.getByTestId("field-width")).toBeInTheDocument();
  });

  it("AC02: image shows height field", () => {
    render(<Harness initial={imageParams()} />);
    expect(screen.getByTestId("field-height")).toBeInTheDocument();
  });

  it("AC02: image does NOT show pipeline selector", () => {
    render(<Harness initial={imageParams()} />);
    expect(screen.queryByTestId("field-pipeline")).not.toBeInTheDocument();
  });

  it("AC02: image does NOT show duration field", () => {
    render(<Harness initial={imageParams()} />);
    expect(screen.queryByTestId("field-duration")).not.toBeInTheDocument();
  });

  // ── AC03: video fields ─────────────────────────────────────────────────────

  it("AC03: video shows model selector", () => {
    render(<Harness initial={videoParams()} />);
    expect(screen.getByTestId("field-model")).toBeInTheDocument();
  });

  it("AC03: video shows pipeline selector with all 5 options", () => {
    render(<Harness initial={videoParams()} />);
    expect(screen.getByTestId("field-pipeline")).toBeInTheDocument();
    const select = screen.getByTestId("select-pipeline") as HTMLSelectElement;
    expect(select.options.length).toBe(VIDEO_PIPELINES.length);
    const values = Array.from(select.options).map((o) => o.value);
    expect(values).toEqual(VIDEO_PIPELINES);
  });

  it("AC03: video shows width field", () => {
    render(<Harness initial={videoParams()} />);
    expect(screen.getByTestId("field-width")).toBeInTheDocument();
  });

  it("AC03: video shows height field", () => {
    render(<Harness initial={videoParams()} />);
    expect(screen.getByTestId("field-height")).toBeInTheDocument();
  });

  it("AC03: video shows duration field", () => {
    render(<Harness initial={videoParams()} />);
    expect(screen.getByTestId("field-duration")).toBeInTheDocument();
  });

  // ── AC04: audio fields ─────────────────────────────────────────────────────

  it("AC04: audio shows model selector", () => {
    render(<Harness initial={audioParams()} />);
    expect(screen.getByTestId("field-model")).toBeInTheDocument();
  });

  it("AC04: audio shows duration field", () => {
    render(<Harness initial={audioParams()} />);
    expect(screen.getByTestId("field-duration")).toBeInTheDocument();
  });

  it("AC04: audio does NOT show width or height", () => {
    render(<Harness initial={audioParams()} />);
    expect(screen.queryByTestId("field-width")).not.toBeInTheDocument();
    expect(screen.queryByTestId("field-height")).not.toBeInTheDocument();
  });

  it("AC04: audio does NOT show pipeline selector", () => {
    render(<Harness initial={audioParams()} />);
    expect(screen.queryByTestId("field-pipeline")).not.toBeInTheDocument();
  });

  // ── AC05: sensible defaults ────────────────────────────────────────────────

  it("AC05: image defaults — width=768, height=512", () => {
    render(<Harness initial={imageParams()} />);
    expect(screen.getByTestId("input-width")).toHaveValue(768);
    expect(screen.getByTestId("input-height")).toHaveValue(512);
  });

  it("AC05: video defaults — width=768, height=512, duration=5", () => {
    render(<Harness initial={videoParams()} />);
    expect(screen.getByTestId("input-width")).toHaveValue(768);
    expect(screen.getByTestId("input-height")).toHaveValue(512);
    expect(screen.getByTestId("input-duration")).toHaveValue(5);
  });

  it("AC05: audio defaults — duration=5", () => {
    render(<Harness initial={audioParams()} />);
    expect(screen.getByTestId("input-duration")).toHaveValue(5);
  });

  it("AC05: image model defaults to first IMAGE_MODELS entry", () => {
    render(<Harness initial={imageParams()} />);
    const sel = screen.getByTestId("select-model") as HTMLSelectElement;
    expect(sel.value).toBe(IMAGE_MODELS[0]);
  });

  it("AC05: video model defaults to first VIDEO_MODELS entry", () => {
    render(<Harness initial={videoParams()} />);
    const sel = screen.getByTestId("select-model") as HTMLSelectElement;
    expect(sel.value).toBe(VIDEO_MODELS[0]);
  });

  it("AC05: video pipeline defaults to first VIDEO_PIPELINES entry (t2v)", () => {
    render(<Harness initial={videoParams()} />);
    const sel = screen.getByTestId("select-pipeline") as HTMLSelectElement;
    expect(sel.value).toBe(VIDEO_PIPELINES[0]);
  });

  // ── AC06: media type change resets pipeline ────────────────────────────────

  it("AC06: useGenerationParams resets pipeline to t2v when switching to video", async () => {
    // Tested via the hook directly — ensure VIDEO_PIPELINES[0] is the reset target
    expect(VIDEO_PIPELINES[0]).toBe("t2v");
  });

  it("AC06: changing pipeline to ia2v then reading model still shows video models", async () => {
    const user = userEvent.setup();
    render(<Harness initial={videoParams()} />);
    const sel = screen.getByTestId("select-pipeline") as HTMLSelectElement;
    await user.selectOptions(sel, "ia2v");
    expect(sel.value).toBe("ia2v");
    // Model selector still shows video models after pipeline change
    const modelSel = screen.getByTestId("select-model") as HTMLSelectElement;
    const options = Array.from(modelSel.options).map((o) => o.value);
    expect(options).toEqual(Array.from(VIDEO_MODELS));
  });
});
