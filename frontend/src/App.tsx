import { useState, useCallback, useRef, useEffect } from "react";
import { useGenerationParams } from "./hooks/useGenerationParams";
import { MediaTypeSelector } from "./components/MediaTypeSelector";
import { ParameterPanel } from "./components/ParameterPanel";
import { ImageUpload } from "./components/ImageUpload";
import { ChatBubble } from "./components/ChatBubble";
import { requiresInputImage } from "./types";
import type { ChatMessage } from "./types";
import "./App.css";

declare const __PARALLAX_API_URL__: string;

let _msgCounter = 0;
function nextId() {
  return `msg-${++_msgCounter}`;
}

export function App() {
  const { params, setMediaType, updateParam } = useGenerationParams();
  const [prompt, setPrompt] = useState("");
  const [inputImage, setInputImage] = useState<File | null>(null);
  const [imageError, setImageError] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const esRef = useRef<EventSource | null>(null);
  const messagesEndRef = useRef<HTMLDivElement | null>(null);

  const needsImage = requiresInputImage(params);

  // Scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Cleanup EventSource on unmount
  useEffect(() => {
    return () => {
      esRef.current?.close();
    };
  }, []);

  const handleImageChange = useCallback((file: File | null) => {
    setInputImage(file);
    if (file) setImageError(null);
  }, []);

  const updateAssistantBubble = useCallback(
    (id: string, patch: Partial<ChatMessage>) => {
      setMessages((prev) =>
        prev.map((m) => (m.id === id ? { ...m, ...patch } : m))
      );
    },
    []
  );

  const handleSubmit = useCallback(() => {
    if (isSubmitting) return;

    if (needsImage && !inputImage) {
      setImageError("Please select a reference image before generating.");
      return;
    }
    setImageError(null);

    const trimmed = prompt.trim();

    // Add user bubble
    const userMsgId = nextId();
    const userMsg: ChatMessage = {
      id: userMsgId,
      role: "user",
      content: trimmed || "(no prompt)",
      status: "complete",
    };

    // Add assistant bubble (streaming)
    const assistantMsgId = nextId();
    const assistantMsg: ChatMessage = {
      id: assistantMsgId,
      role: "assistant",
      content: "Starting…",
      status: "streaming",
      progress: 0,
    };

    setMessages((prev) => [...prev, userMsg, assistantMsg]);
    setIsSubmitting(true);
    setPrompt("");

    const formData = new FormData();
    if (inputImage) formData.append("image", inputImage);
    formData.append("media_type", params.mediaType);
    formData.append("model", params.model);
    formData.append("pipeline", params.pipeline);
    formData.append("width", String(params.width));
    formData.append("height", String(params.height));
    formData.append("duration", String(params.duration));
    formData.append("prompt", trimmed);

    fetch(`${__PARALLAX_API_URL__}/create/${params.mediaType}`, {
      method: "POST",
      body: formData,
    })
      .then(async (res) => {
        if (!res.ok) {
          const body = await res.text().catch(() => "");
          throw new Error(`Server error ${res.status}${body ? `: ${body}` : ""}`);
        }
        return res.json();
      })
      .then((data: { job_id: string }) => {
        const jobId = data.job_id;
        updateAssistantBubble(assistantMsgId, {
          content: "Job queued — connecting…",
        });

        const es = new EventSource(
          `${__PARALLAX_API_URL__}/jobs/${jobId}/stream`
        );
        esRef.current = es;

        let sseTimeoutId: ReturnType<typeof setTimeout> | null = null;
        const clearSseTimeout = () => {
          if (sseTimeoutId !== null) {
            clearTimeout(sseTimeoutId);
            sseTimeoutId = null;
          }
        };
        const armSseTimeout = () => {
          clearSseTimeout();
          sseTimeoutId = setTimeout(() => {
            updateAssistantBubble(assistantMsgId, {
              content:
                "No response from server for 30 seconds — generation may have stalled.",
              status: "timeout",
            });
            es.close();
            setIsSubmitting(false);
          }, 30_000);
        };
        armSseTimeout();

        es.onmessage = (event) => {
          armSseTimeout();
          try {
            const progress = JSON.parse(event.data) as {
              step: string;
              pct: number;
              error?: string | null;
            };
            const pct = Math.round((progress.pct ?? 0) * 100);

            if (progress.step === "done") {
              clearSseTimeout();
              const extMap: Record<string, string> = {
                image: "png",
                video: "mp4",
                audio: "wav",
              };
              const ext = extMap[params.mediaType] ?? "bin";
              const mediaUrl = `${__PARALLAX_API_URL__}/jobs/${jobId}/result`;
              updateAssistantBubble(assistantMsgId, {
                content: "Generation complete ✓",
                status: "complete",
                progress: 100,
                progressLabel: "100%",
                mediaUrl,
                mediaType: params.mediaType,
                filename: `parallax-${jobId}.${ext}`,
              });
              es.close();
              setIsSubmitting(false);
            } else if (progress.step === "error") {
              clearSseTimeout();
              updateAssistantBubble(assistantMsgId, {
                content: progress.error ?? "Generation failed.",
                status: "error",
              });
              es.close();
              setIsSubmitting(false);
            } else {
              updateAssistantBubble(assistantMsgId, {
                content: progress.step,
                status: "streaming",
                progress: pct,
                progressLabel: `${pct}%`,
              });
            }
          } catch {
            // Ignore malformed events
          }
        };

        es.onerror = () => {
          clearSseTimeout();
          updateAssistantBubble(assistantMsgId, {
            content: "Connection lost — check job status.",
            status: "connection-lost",
          });
          es.close();
          setIsSubmitting(false);
        };
      })
      .catch((err: unknown) => {
        updateAssistantBubble(assistantMsgId, {
          content:
            err instanceof Error ? err.message : "Request failed.",
          status: "error",
        });
        setIsSubmitting(false);
      });
  }, [
    isSubmitting,
    needsImage,
    inputImage,
    params,
    prompt,
    updateAssistantBubble,
  ]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleSubmit();
      }
    },
    [handleSubmit]
  );

  return (
    <div className="app">
      <header className="app-header">
        <span className="app-title">Parallax</span>
      </header>

      <main className="chat-area">
        {messages.length === 0 ? (
          <div className="chat-empty" data-testid="chat-empty">
            <p className="chat-empty-hint">
              Type a prompt and press Generate to begin.
            </p>
          </div>
        ) : (
          <div className="chat-messages" data-testid="chat-messages" role="log" aria-label="Chat messages">
            {messages.map((msg) => (
              <ChatBubble key={msg.id} message={msg} />
            ))}
            <div ref={messagesEndRef} />
          </div>
        )}
      </main>

      <footer className="composer">
        <div className="composer-config">
          <MediaTypeSelector value={params.mediaType} onChange={setMediaType} />
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
            data-testid="prompt-input"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={isSubmitting}
          />
          <button
            className="composer-send"
            type="button"
            aria-label="Generate"
            onClick={handleSubmit}
            disabled={isSubmitting}
          >
            ➤
          </button>
        </div>
      </footer>
    </div>
  );
}
