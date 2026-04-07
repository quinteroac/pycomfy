import styles from "./ChatBubble.module.css";
import type { ChatMessage } from "../types";

interface ChatBubbleProps {
  message: ChatMessage;
}

export function ChatBubble({ message }: ChatBubbleProps) {
  const { role, content, status, progress, progressLabel } = message;
  const isAssistant = role === "assistant";
  const isStreaming = status === "streaming";

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
    </div>
  );
}
