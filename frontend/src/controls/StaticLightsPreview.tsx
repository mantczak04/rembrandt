import type { LightConfig } from "../types";
import styles from "./Controls.module.css";

export type StaticLightsPreviewProps = {
  lights: LightConfig[];
};

function formatVec3(value: [number, number, number] | undefined): string {
  if (!value) {
    return "[?, ?, ?]";
  }
  return `[${value.map((n) => n.toFixed(2)).join(", ")}]`;
}

export default function StaticLightsPreview({ lights }: StaticLightsPreviewProps) {
  return (
    <div className={styles.readOnlyBlock}>
      <p className={styles.hint}>
        Static lights are edited in YAML only. When light randomization is{" "}
        <code>random</code>, this list is ignored at render time.
      </p>
      {lights.length === 0 ? (
        <p className={styles.hint}>No static lights configured.</p>
      ) : (
        <ul className={styles.readOnlyList}>
          {lights.map((light, index) => (
            <li key={index}>
              <strong>{light.light_type ?? "POINT"}</strong>
              {" · "}
              loc {formatVec3(light.location)}
              {" · "}
              look_at {formatVec3(light.look_at)}
              {light.energy !== null && light.energy !== undefined
                ? ` · energy ${light.energy}`
                : ""}
              {light.size !== undefined ? ` · size ${light.size}` : ""}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
