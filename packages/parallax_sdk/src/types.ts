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
