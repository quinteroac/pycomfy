/** Media types supported by the Parallax server. */
export type MediaType = "image" | "video" | "audio";

/** Video pipeline types. */
export type VideoPipeline = "t2v" | "i2v" | "is2v" | "flf2v" | "ia2v";

/** All generation parameters in a single flat shape. */
export interface GenerationParams {
  mediaType: MediaType;
  model: string;
  pipeline: VideoPipeline;
  width: number;
  height: number;
  duration: number;
}

export const IMAGE_MODELS = ["sdxl", "flux_klein", "anima", "z_image"] as const;
export const VIDEO_MODELS = ["ltx23", "ltx2", "wan21", "wan22"] as const;
export const AUDIO_MODELS = ["ace_step"] as const;

export const VIDEO_PIPELINES: VideoPipeline[] = [
  "t2v",
  "i2v",
  "is2v",
  "flf2v",
  "ia2v",
];

/** Pipelines that require a reference input image. */
export const IMAGE_REQUIRED_PIPELINES: ReadonlyArray<VideoPipeline> = [
  "i2v",
  "is2v",
  "ia2v",
];

export function requiresInputImage(
  params: Pick<GenerationParams, "mediaType" | "pipeline">
): boolean {
  return (
    params.mediaType === "video" &&
    (IMAGE_REQUIRED_PIPELINES as string[]).includes(params.pipeline)
  );
}

export const PIPELINE_LABELS: Record<VideoPipeline, string> = {
  t2v: "Text-to-Video",
  i2v: "Image-to-Video",
  is2v: "Image+Start-to-Video",
  flf2v: "First/Last-to-Video",
  ia2v: "Image+Audio-to-Video",
};

export const DEFAULT_PARAMS: GenerationParams = {
  mediaType: "image",
  model: IMAGE_MODELS[0],
  pipeline: VIDEO_PIPELINES[0],
  width: 768,
  height: 512,
  duration: 5,
};

export function modelsForType(type: MediaType): readonly string[] {
  if (type === "image") return IMAGE_MODELS;
  if (type === "video") return VIDEO_MODELS;
  return AUDIO_MODELS;
}
