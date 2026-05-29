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

  const updateLow = (next: number) => {
    onChange([clamp(next, min, high), high]);
  };

  const updateHigh = (next: number) => {
    onChange([low, clamp(next, low, max)]);
  };

  return (
    <fieldset className={styles.fieldset}>
      <legend className={styles.legend}>{label}</legend>
      <div className={styles.rangeRow}>
        <label className={styles.label} htmlFor={`${id}-min`}>
          Min
        </label>
        <input
          id={`${id}-min`}
          className={styles.input}
          type="number"
          min={min}
          max={max}
          step={step}
          value={low}
          onChange={(event) => updateLow(Number(event.target.value))}
        />
        <label className={styles.label} htmlFor={`${id}-max`}>
          Max
        </label>
        <input
          id={`${id}-max`}
          className={styles.input}
          type="number"
          min={min}
          max={max}
          step={step}
          value={high}
          onChange={(event) => updateHigh(Number(event.target.value))}
        />
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
