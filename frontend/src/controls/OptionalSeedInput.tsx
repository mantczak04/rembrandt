import { clampNumber } from "./formUtils";
import styles from "./Controls.module.css";

export type OptionalSeedInputProps = {
  id: string;
  label: string;
  seed: number | null | undefined;
  defaultSeed?: number;
  onChange: (seed: number | null) => void;
};

export default function OptionalSeedInput({
  id,
  label,
  seed,
  defaultSeed = 42,
  onChange,
}: OptionalSeedInputProps) {
  const enabled = seed !== null && seed !== undefined;

  return (
    <>
      <label className={styles.checkboxRow}>
        <input
          type="checkbox"
          checked={enabled}
          onChange={(event) =>
            onChange(event.target.checked ? (seed ?? defaultSeed) : null)
          }
        />
        {label}
      </label>
      {enabled ? (
        <div className={styles.row}>
          <label className={styles.label} htmlFor={id}>
            Seed value
          </label>
          <input
            id={id}
            className={styles.input}
            type="number"
            min={0}
            step={1}
            value={seed}
            onChange={(event) =>
              onChange(Math.round(clampNumber(Number(event.target.value), 0, 2 ** 32)))
            }
          />
        </div>
      ) : null}
    </>
  );
}
