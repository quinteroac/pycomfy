import { useGenerationParams } from "./hooks/useGenerationParams";
import { MediaTypeSelector } from "./components/MediaTypeSelector";
import { ParameterPanel } from "./components/ParameterPanel";
import "./App.css";

export function App() {
  const { params, setMediaType, updateParam } = useGenerationParams();

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
          <textarea
            className="composer-textarea"
            placeholder="Describe what you want to generate…"
            rows={1}
            aria-label="Prompt"
          />
          <button className="composer-send" type="button" aria-label="Generate">
            ➤
          </button>
        </div>
      </footer>
    </div>
  );
}
