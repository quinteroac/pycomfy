import styles from "./ChatBubble.module.css";
import type { ChatMessage } from "../types";

interface ChatBubbleProps {
  message: ChatMessage;
}

export function ChatBubble({ message }: ChatBubbleProps) {
  const { role, content, status, progress, progressLabel, mediaUrl, mediaType, filename } = message;
  const isAssistant = role === "assistant";
  const isStreaming = status === "streaming";
  const isComplete = status === "complete";

  const wrapperClass = [
    styles.bubble,
    isAssistant ? styles["bubble-assistant"] : styles["bubble-user"],
    status === "error" ? styles["status-error"] : "",
    status === "connection-lost" ? styles["status-connection-lost"] : "",
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <div className={wrapperClass} data-testid={`bubble-${role}`}>
      <div className={styles["bubble-content"]}>
        {content}

        {isAssistant && isStreaming && (
          <div className={styles["progress-row"]} data-testid="progress-row">
            <span className={styles.spinner} aria-hidden="true" />
            {typeof progress === "number" && (
              <div className={styles["progress-bar-wrap"]}>
                <div
                  className={styles["progress-bar-fill"]}
                  style={{ width: `${progress}%` }}
                  data-testid="progress-bar"
                />
              </div>
            )}
            {progressLabel && (
              <span
                className={styles["progress-label"]}
                data-testid="progress-label"
              >
                {progressLabel}
              </span>
            )}
          </div>
        )}
      </div>

      {isAssistant && isComplete && mediaUrl && (
        <div className={styles["media-container"]} data-testid="media-container">
          {mediaType === "image" && (
            <img
              src={mediaUrl}
              alt="Generated image"
              className={styles["media-image"]}
              data-testid="media-image"
            />
          )}
          {mediaType === "video" && (
            <video
              src={mediaUrl}
              controls
              className={styles["media-video"]}
              data-testid="media-video"
            />
          )}
          {mediaType === "audio" && (
            <audio
              src={mediaUrl}
              controls
              className={styles["media-audio"]}
              data-testid="media-audio"
            />
          )}
          <a
            href={mediaUrl}
            download={filename ?? "output"}
            className={styles["download-btn"]}
            data-testid="download-btn"
          >
            Download
          </a>
        </div>
      )}
    </div>
  );
}
