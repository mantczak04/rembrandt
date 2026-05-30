import { useState } from "react";
import styles from "./Controls.module.css";

export type RangeInputProps = {
  id: string;
  label: string;
  min: number;
  max: number;
  step?: number;
  value: [number, number];
  onChange: (value: [number, number]) => void;
};

export default function RangeInput({
  id,
  label,
  min,
  max,
  step = 1,
  value,
  onChange,
}: RangeInputProps) {
  const [low, high] = value;
  const [activeThumb, setActiveThumb] = useState<"min" | "max" | null>(null);

  const span = max - min;
  const lowPercent = span === 0 ? 0 : ((low - min) / span) * 100;
  const highPercent = span === 0 ? 100 : ((high - min) / span) * 100;

  const updateLow = (next: number) => {
    onChange([clamp(next, min, high), high]);
  };

  const updateHigh = (next: number) => {
    onChange([low, clamp(next, low, max)]);
  };

  const formatValue = (n: number) =>
    step >= 1 ? String(Math.round(n)) : n.toFixed(1);

  return (
    <fieldset className={styles.fieldset}>
      <legend className={styles.legend}>{label}</legend>
      <div className={styles.rangeSlider}>
        <div className={styles.rangeTrack} aria-hidden="true">
          <div
            className={styles.rangeFill}
            style={{
              left: `${lowPercent}%`,
              width: `${highPercent - lowPercent}%`,
            }}
          />
        </div>
        <input
          id={`${id}-min`}
          className={styles.rangeThumb}
          type="range"
          min={min}
          max={max}
          step={step}
          value={low}
          aria-label={`${label} minimum`}
          style={{ zIndex: activeThumb === "min" ? 3 : 2 }}
          onPointerDown={() => setActiveThumb("min")}
          onPointerUp={() => setActiveThumb(null)}
          onBlur={() => setActiveThumb(null)}
          onChange={(event) => updateLow(Number(event.target.value))}
        />
        <input
          id={`${id}-max`}
          className={styles.rangeThumb}
          type="range"
          min={min}
          max={max}
          step={step}
          value={high}
          aria-label={`${label} maximum`}
          style={{ zIndex: activeThumb === "max" ? 3 : 1 }}
          onPointerDown={() => setActiveThumb("max")}
          onPointerUp={() => setActiveThumb(null)}
          onBlur={() => setActiveThumb(null)}
          onChange={(event) => updateHigh(Number(event.target.value))}
        />
      </div>
      <div className={styles.rangeValues} aria-live="polite">
        <span className={styles.rangeValue}>
          <span className={styles.rangeValueLabel}>Min</span>
          {formatValue(low)}
        </span>
        <span className={styles.rangeValue}>
          <span className={styles.rangeValueLabel}>Max</span>
          {formatValue(high)}
        </span>
      </div>
    </fieldset>
  );
}

function clamp(value: number, min: number, max: number): number {
  if (!Number.isFinite(value)) {
    return min;
  }
  return Math.min(Math.max(value, min), max);
}
