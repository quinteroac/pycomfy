// Shared request/response types across CLI, ms, and mcp

export interface GenerateImageRequest {
  prompt: string;
  negative_prompt?: string;
  width?: number;
  height?: number;
  steps?: number;
}

export interface EditImageRequest {
  image_path: string;
  prompt: string;
  steps?: number;
}

export interface GenerateImageResponse {
  image_path: string;
  seed: number;
}

// --- Video placeholders ---

export interface GenerateVideoRequest {
  prompt: string;
  negative_prompt?: string;
  width?: number;
  height?: number;
  frames?: number;
  fps?: number;
  steps?: number;
}

export interface GenerateVideoResponse {
  video_path: string;
  seed: number;
}

// --- Audio placeholders ---

export interface GenerateAudioRequest {
  prompt: string;
  negative_prompt?: string;
  duration_seconds?: number;
  steps?: number;
}

export interface GenerateAudioResponse {
  audio_path: string;
  seed: number;
}
