
import { renderHook, act } from "@testing-library/react";
import { useGenerationParams } from "../hooks/useGenerationParams";
import {
  DEFAULT_PARAMS,
  IMAGE_MODELS,
  VIDEO_MODELS,
  AUDIO_MODELS,
  VIDEO_PIPELINES,
} from "../types";

describe("useGenerationParams — US-001", () => {
  it("initialises with DEFAULT_PARAMS", () => {
    const { result } = renderHook(() => useGenerationParams());
    expect(result.current.params).toEqual(DEFAULT_PARAMS);
  });

  it("AC05: default mediaType is image", () => {
    const { result } = renderHook(() => useGenerationParams());
    expect(result.current.params.mediaType).toBe("image");
  });

  it("AC05: default model is first IMAGE_MODELS entry", () => {
    const { result } = renderHook(() => useGenerationParams());
    expect(result.current.params.model).toBe(IMAGE_MODELS[0]);
  });

  it("AC05: default width=768, height=512, duration=5", () => {
    const { result } = renderHook(() => useGenerationParams());
    expect(result.current.params.width).toBe(768);
    expect(result.current.params.height).toBe(512);
    expect(result.current.params.duration).toBe(5);
  });

  it("AC06: switching to video resets model to first VIDEO model", () => {
    const { result } = renderHook(() => useGenerationParams());
    act(() => result.current.setMediaType("video"));
    expect(result.current.params.mediaType).toBe("video");
    expect(result.current.params.model).toBe(VIDEO_MODELS[0]);
  });

  it("AC06: switching to audio resets model to first AUDIO model", () => {
    const { result } = renderHook(() => useGenerationParams());
    act(() => result.current.setMediaType("audio"));
    expect(result.current.params.model).toBe(AUDIO_MODELS[0]);
  });

  it("AC06: switching media type always resets pipeline to t2v", () => {
    const { result } = renderHook(() => useGenerationParams());
    // first set pipeline to ia2v
    act(() => result.current.setMediaType("video"));
    act(() => result.current.updateParam("pipeline", "ia2v"));
    expect(result.current.params.pipeline).toBe("ia2v");
    // switch back to image and then to video — pipeline must reset
    act(() => result.current.setMediaType("image"));
    act(() => result.current.setMediaType("video"));
    expect(result.current.params.pipeline).toBe(VIDEO_PIPELINES[0]);
  });

  it("updateParam updates a single field", () => {
    const { result } = renderHook(() => useGenerationParams());
    act(() => result.current.updateParam("width", 1024));
    expect(result.current.params.width).toBe(1024);
  });
});
