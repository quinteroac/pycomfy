import { MediaType } from "../types";
import styles from "./MediaTypeSelector.module.css";

interface Props {
  value: MediaType;
  onChange: (type: MediaType) => void;
}

const MEDIA_TYPES: { value: MediaType; label: string; icon: string }[] = [
  { value: "image", label: "Image", icon: "⬚" },
  { value: "video", label: "Video", icon: "▶" },
  { value: "audio", label: "Audio", icon: "♪" },
];

export function MediaTypeSelector({ value, onChange }: Props) {
  return (
    <div className={styles.root} role="group" aria-label="Media type">
      {MEDIA_TYPES.map(({ value: type, label, icon }) => (
        <button
          key={type}
          type="button"
          role="radio"
          aria-checked={value === type}
          aria-label={label}
          className={`${styles.button} ${value === type ? styles.active : ""}`}
          onClick={() => onChange(type)}
          data-testid={`media-type-${type}`}
        >
          <span className={styles.icon} aria-hidden="true">
            {icon}
          </span>
          <span className={styles.label}>{label}</span>
        </button>
      ))}
    </div>
  );
}
