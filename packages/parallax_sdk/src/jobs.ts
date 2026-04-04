// Job types shared across parallax packages

export interface ParallaxJobData {
  action: string;
  media: string;
  model: string;
  script: string;
  args: string[];
  scriptBase: string;
  uvPath: string;
}

export interface ParallaxJobResult {
  outputPath: string;
}

export interface PythonProgress {
  step: string;
  pct: number;
  frame?: number;
  total?: number;
  output?: string;
  error?: string;
}
