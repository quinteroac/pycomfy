import { useState, useCallback } from "react";
import { useGenerationParams } from "./hooks/useGenerationParams";
import { MediaTypeSelector } from "./components/MediaTypeSelector";
import { ParameterPanel } from "./components/ParameterPanel";
import { ImageUpload } from "./components/ImageUpload";
import { requiresInputImage } from "./types";
import "./App.css";

declare const __PARALLAX_API_URL__: string;

export function App() {
  const { params, setMediaType, updateParam } = useGenerationParams();
  const [prompt, setPrompt] = useState("");
  const [inputImage, setInputImage] = useState<File | null>(null);
  const [imageError, setImageError] = useState<string | null>(null);

  const needsImage = requiresInputImage(params);

  const handleImageChange = useCallback((file: File | null) => {
    setInputImage(file);
    if (file) setImageError(null);
  }, []);

  const handleSubmit = useCallback(() => {
    if (needsImage && !inputImage) {
      setImageError("Please select a reference image before generating.");
      return;
    }
    setImageError(null);

    const formData = new FormData();
    if (inputImage) formData.append("image", inputImage);
    formData.append("media_type", params.mediaType);
    formData.append("model", params.model);
    formData.append("pipeline", params.pipeline);
    formData.append("width", String(params.width));
    formData.append("height", String(params.height));
    formData.append("duration", String(params.duration));
    formData.append("prompt", prompt);

    fetch(`${__PARALLAX_API_URL__}/create/${params.mediaType}`, {
      method: "POST",
      body: formData,
    });
  }, [needsImage, inputImage, params, prompt]);

  return (
    <div className="app">
      <header className="app-header">
        <span className="app-logo">◈</span>
        <span className="app-title">Parallax</span>
      </header>

      <main className="chat-area">
        <div className="chat-empty">
          <p className="chat-empty-hint">
            Select a media type and configure parameters below, then type your
            prompt to begin.
          </p>
        </div>
      </main>

      <footer className="composer">
        <div className="composer-config">
          <MediaTypeSelector value={params.mediaType} onChange={setMediaType} />
          <div className="composer-divider" aria-hidden="true" />
          <ParameterPanel params={params} onUpdate={updateParam} />
        </div>

        <div className="composer-input-row">
          {needsImage && (
            <ImageUpload
              value={inputImage}
              onChange={handleImageChange}
              error={imageError}
            />
          )}
          <textarea
            className="composer-textarea"
            placeholder="Describe what you want to generate…"
            rows={1}
            aria-label="Prompt"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
          />
          <button
            className="composer-send"
            type="button"
            aria-label="Generate"
            onClick={handleSubmit}
          >
            ➤
          </button>
        </div>
      </footer>
    </div>
  );
}
