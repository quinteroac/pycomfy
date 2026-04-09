import { useRef, useEffect, useState } from "react";
import styles from "./ImageUpload.module.css";

const ACCEPTED_TYPES = ".jpg,.jpeg,.png,.webp";

interface Props {
  value: File | null;
  onChange: (file: File | null) => void;
  error?: string | null;
}

const INPUT_ID = "image-upload-file-input";

export function ImageUpload({ value, onChange, error }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  useEffect(() => {
    if (!value) {
      setPreviewUrl(null);
      return;
    }
    const url = URL.createObjectURL(value);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [value]);

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0] ?? null;
    onChange(file);
  }

  function handleRemove() {
    onChange(null);
    if (inputRef.current) inputRef.current.value = "";
  }

  return (
    <div className={styles.root} data-testid="image-upload">
      <input
        ref={inputRef}
        id={INPUT_ID}
        type="file"
        accept={ACCEPTED_TYPES}
        className={styles.hiddenInput}
        data-testid="image-upload-input"
        onChange={handleFileChange}
      />

      {previewUrl ? (
        <div className={styles.preview} data-testid="image-preview-container">
          <img
            src={previewUrl}
            alt="Reference image preview"
            className={styles.thumbnail}
            data-testid="image-thumbnail"
          />
          <button
            type="button"
            className={styles.removeBtn}
            aria-label="Remove image"
            data-testid="image-upload-remove"
            onClick={handleRemove}
          >
            ✕
          </button>
        </div>
      ) : (
        <label
          htmlFor={INPUT_ID}
          className={`${styles.uploadBtn}${error ? ` ${styles.uploadBtnError}` : ""}`}
          data-testid="image-upload-trigger"
        >
          <span className={styles.uploadIcon} aria-hidden="true">🖼</span>
          <span>Reference image</span>
        </label>
      )}

      {error && (
        <p className={styles.errorMsg} data-testid="image-upload-error" role="alert">
          {error}
        </p>
      )}
    </div>
  );
}
