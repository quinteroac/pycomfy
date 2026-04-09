import {
  GenerationParams,
  MediaType,
  PIPELINE_LABELS,
  VIDEO_PIPELINES,
  VideoPipeline,
  modelsForType,
} from "../types";
import styles from "./ParameterPanel.module.css";

interface Props {
  params: GenerationParams;
  onUpdate: <K extends keyof GenerationParams>(
    key: K,
    value: GenerationParams[K]
  ) => void;
}

function Field({
  label,
  children,
  testId,
}: {
  label: string;
  children: React.ReactNode;
  testId?: string;
}) {
  return (
    <label className={styles.field} data-testid={testId}>
      <span className={styles.fieldLabel}>{label}</span>
      {children}
    </label>
  );
}

function ModelSelect({
  mediaType,
  value,
  onChange,
}: {
  mediaType: MediaType;
  value: string;
  onChange: (v: string) => void;
}) {
  const models = modelsForType(mediaType);
  return (
    <Field label="Model" testId="field-model">
      <select
        className={styles.select}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        aria-label="Model"
        data-testid="select-model"
      >
        {models.map((m) => (
          <option key={m} value={m}>
            {m}
          </option>
        ))}
      </select>
    </Field>
  );
}

function NumberInput({
  label,
  value,
  min,
  max,
  step,
  testId,
  onChange,
}: {
  label: string;
  value: number;
  min?: number;
  max?: number;
  step?: number;
  testId?: string;
  onChange: (v: number) => void;
}) {
  return (
    <Field label={label} testId={testId}>
      <input
        type="number"
        className={styles.input}
        value={value}
        min={min}
        max={max}
        step={step ?? 1}
        aria-label={label}
        data-testid={`input-${label.toLowerCase()}`}
        onChange={(e) => {
          const parsed = parseFloat(e.target.value);
          if (!isNaN(parsed)) onChange(parsed);
        }}
      />
    </Field>
  );
}

export function ParameterPanel({ params, onUpdate }: Props) {
  const { mediaType, model, pipeline, width, height, duration } = params;

  return (
    <div className={styles.root} data-testid="parameter-panel">
      <ModelSelect
        mediaType={mediaType}
        value={model}
        onChange={(v) => onUpdate("model", v)}
      />

      {mediaType === "video" && (
        <Field label="Pipeline" testId="field-pipeline">
          <select
            className={styles.select}
            value={pipeline}
            onChange={(e) => onUpdate("pipeline", e.target.value as VideoPipeline)}
            aria-label="Pipeline type"
            data-testid="select-pipeline"
          >
            {VIDEO_PIPELINES.map((p) => (
              <option key={p} value={p}>
                {PIPELINE_LABELS[p]}
              </option>
            ))}
          </select>
        </Field>
      )}

      {(mediaType === "image" || mediaType === "video") && (
        <>
          <NumberInput
            label="Width"
            value={width}
            min={64}
            max={2048}
            step={64}
            testId="field-width"
            onChange={(v) => onUpdate("width", v)}
          />
          <NumberInput
            label="Height"
            value={height}
            min={64}
            max={2048}
            step={64}
            testId="field-height"
            onChange={(v) => onUpdate("height", v)}
          />
        </>
      )}

      {(mediaType === "video" || mediaType === "audio") && (
        <NumberInput
          label="Duration"
          value={duration}
          min={1}
          max={120}
          step={1}
          testId="field-duration"
          onChange={(v) => onUpdate("duration", v)}
        />
      )}
    </div>
  );
}
