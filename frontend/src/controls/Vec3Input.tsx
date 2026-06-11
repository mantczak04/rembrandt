import type { Vec3 } from "../types";
import { clampNumber } from "./formUtils";
import styles from "./Controls.module.css";

export type Vec3InputProps = {
  id: string;
  label: string;
  value: Vec3;
  min?: number;
  max?: number;
  step?: number;
  onChange: (value: Vec3) => void;
};

const AXES: Array<{ index: 0 | 1 | 2; label: string }> = [
  { index: 0, label: "R" },
  { index: 1, label: "G" },
  { index: 2, label: "B" },
];

export default function Vec3Input({
  id,
  label,
  value,
  min = 0,
  max = 1,
  step = 0.01,
  onChange,
}: Vec3InputProps) {
  const updateAxis = (index: 0 | 1 | 2, next: number) => {
    const copy: Vec3 = [...value];
    copy[index] = clampNumber(next, min, max);
    onChange(copy);
  };

  return (
    <fieldset className={styles.fieldset}>
      <legend className={styles.legend}>{label}</legend>
      <div className={styles.vec3Row}>
        {AXES.map((axis) => (
          <div key={axis.index} className={styles.vec3Field}>
            <label className={styles.label} htmlFor={`${id}-${axis.index}`}>
              {axis.label}
            </label>
            <input
              id={`${id}-${axis.index}`}
              className={styles.input}
              type="number"
              min={min}
              max={max}
              step={step}
              value={value[axis.index]}
              onChange={(event) => updateAxis(axis.index, Number(event.target.value))}
            />
          </div>
        ))}
      </div>
    </fieldset>
  );
}
