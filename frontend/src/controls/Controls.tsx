import type { CameraConfig, RembrandtConfig, SamplingStrategy } from "../types";
import RangeInput from "./RangeInput";
import styles from "./Controls.module.css";

function clampNumber(value: number, min: number, max: number): number {
  if (!Number.isFinite(value)) {
    return min;
  }
  return Math.min(Math.max(value, min), max);
}

export type ControlsProps = {
  config: RembrandtConfig;
  objectPathInput: string;
  showCameras: boolean;
  showRays: boolean;
  loadingMesh: boolean;
  loadingPoses: boolean;
  meshError: string | null;
  posesError: string | null;
  onObjectPathInputChange: (path: string) => void;
  onLoadMesh: () => void;
  onCameraChange: (camera: CameraConfig) => void;
  onShowCamerasChange: (value: boolean) => void;
  onShowRaysChange: (value: boolean) => void;
};

export default function Controls({
  config,
  objectPathInput,
  showCameras,
  showRays,
  loadingMesh,
  loadingPoses,
  meshError,
  posesError,
  onObjectPathInputChange,
  onLoadMesh,
  onCameraChange,
  onShowCamerasChange,
  onShowRaysChange,
}: ControlsProps) {
  const camera = config.camera;

  const updateCamera = (patch: Partial<CameraConfig>) => {
    onCameraChange({ ...camera, ...patch });
  };

  return (
    <aside className={styles.panel}>
      <h2 className={styles.sectionTitle}>Object</h2>
      <div className={styles.row}>
        <label className={styles.label} htmlFor="object-path">
          OBJ path
        </label>
        <input
          id="object-path"
          className={styles.input}
          type="text"
          value={objectPathInput}
          placeholder="/path/to/model.obj"
          onChange={(event) => onObjectPathInputChange(event.target.value)}
        />
        <button
          type="button"
          className={styles.button}
          disabled={loadingMesh || objectPathInput.trim() === ""}
          onClick={onLoadMesh}
        >
          {loadingMesh ? "Loading…" : "Load mesh"}
        </button>
        {meshError ? <p className={styles.error}>{meshError}</p> : null}
      </div>

      <h2 className={styles.sectionTitle}>Camera sampling</h2>
      <div className={styles.row}>
        <label className={styles.label} htmlFor="pose-count">
          Number of poses
        </label>
        <input
          id="pose-count"
          className={styles.input}
          type="number"
          min={1}
          max={500}
          step={1}
          value={camera.n}
          onChange={(event) =>
            updateCamera({
              n: Math.round(clampNumber(Number(event.target.value), 1, 500)),
            })
          }
        />
      </div>

      <RangeInput
        id="azimuth"
        label="Azimuth range (degrees)"
        min={0}
        max={360}
        step={1}
        value={camera.azimuth_range ?? [0, 360]}
        onChange={(azimuth_range) => updateCamera({ azimuth_range })}
      />

      <RangeInput
        id="elevation"
        label="Elevation range (degrees)"
        min={-90}
        max={90}
        step={1}
        value={camera.elevation_range ?? [-10, 30]}
        onChange={(elevation_range) => updateCamera({ elevation_range })}
      />

      <RangeInput
        id="distance"
        label="Distance range"
        min={0.1}
        max={20}
        step={0.1}
        value={camera.distance_range ?? [3, 5]}
        onChange={(distance_range) => updateCamera({ distance_range })}
      />

      <div className={styles.row}>
        <label className={styles.label} htmlFor="strategy">
          Strategy
        </label>
        <select
          id="strategy"
          className={styles.select}
          value={camera.strategy ?? "random"}
          onChange={(event) =>
            updateCamera({ strategy: event.target.value as SamplingStrategy })
          }
        >
          <option value="random">random</option>
          <option value="fibonacci">fibonacci</option>
        </select>
      </div>

      <label className={styles.checkboxRow}>
        <input
          type="checkbox"
          checked={camera.seed !== null && camera.seed !== undefined}
          onChange={(event) =>
            updateCamera({ seed: event.target.checked ? (camera.seed ?? 42) : null })
          }
        />
        Use seed
      </label>

      {camera.seed !== null && camera.seed !== undefined ? (
        <div className={styles.row}>
          <label className={styles.label} htmlFor="seed">
            Seed
          </label>
          <input
            id="seed"
            className={styles.input}
            type="number"
            min={0}
            step={1}
            value={camera.seed}
            onChange={(event) =>
              updateCamera({
                seed: Math.round(
                  clampNumber(Number(event.target.value), 0, 2 ** 32),
                ),
              })
            }
          />
        </div>
      ) : null}

      <h2 className={styles.sectionTitle}>Preview display</h2>
      <label className={styles.checkboxRow}>
        <input
          type="checkbox"
          checked={showCameras}
          onChange={(event) => onShowCamerasChange(event.target.checked)}
        />
        Show camera markers
      </label>
      <label className={styles.checkboxRow}>
        <input
          type="checkbox"
          checked={showRays}
          onChange={(event) => onShowRaysChange(event.target.checked)}
        />
        Show look-at rays
      </label>

      {loadingPoses ? <p className={styles.hint}>Updating preview…</p> : null}
      {posesError ? <p className={styles.error}>{posesError}</p> : null}
    </aside>
  );
}
