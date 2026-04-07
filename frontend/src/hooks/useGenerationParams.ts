import { useState, useCallback } from "react";
import {
  DEFAULT_PARAMS,
  GenerationParams,
  MediaType,
  VIDEO_PIPELINES,
  modelsForType,
} from "../types";

export function useGenerationParams() {
  const [params, setParams] = useState<GenerationParams>(DEFAULT_PARAMS);

  const setMediaType = useCallback((type: MediaType) => {
    setParams((prev) => ({
      ...prev,
      mediaType: type,
      model: modelsForType(type)[0],
      pipeline: VIDEO_PIPELINES[0],
    }));
  }, []);

  const updateParam = useCallback(
    <K extends keyof GenerationParams>(key: K, value: GenerationParams[K]) => {
      setParams((prev) => ({ ...prev, [key]: value }));
    },
    []
  );

  return { params, setMediaType, updateParam };
}
