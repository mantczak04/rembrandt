import type {
  BackgroundConfig,
  CameraConfig,
  FramingConfig,
  LabelsConfig,
  LightRandomizationConfig,
  LightType,
  OutputConfig,
  RembrandtConfig,
  RenderConfig,
  SamplingStrategy,
  SourceUpAxis,
} from "../types";
import OptionalSeedInput from "./OptionalSeedInput";
import RangeInput from "./RangeInput";
import StaticLightsPreview from "./StaticLightsPreview";
import Vec3Input from "./Vec3Input";
import { clampNumber } from "./formUtils";
import styles from "./Controls.module.css";

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
  onObjectUpAxisChange: (upAxis: SourceUpAxis) => void;
  onObjectNormalizeChange: (normalize: boolean) => void;
  onLoadMesh: () => void;
  onCameraChange: (camera: CameraConfig) => void;
  onConfigChange: (config: RembrandtConfig) => void;
  onShowCamerasChange: (value: boolean) => void;
  onShowRaysChange: (value: boolean) => void;
};

const LIGHT_TYPES: LightType[] = ["POINT", "SUN", "AREA"];

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
  onObjectUpAxisChange,
  onObjectNormalizeChange,
  onLoadMesh,
  onCameraChange,
  onConfigChange,
  onShowCamerasChange,
  onShowRaysChange,
}: ControlsProps) {
  const camera = config.camera;
  const render = config.render ?? {};
  const output = config.output ?? {};
  const background = config.background ?? {};
  const lightRandomization = config.light_randomization ?? {};
  const labels = config.labels ?? {};
  const framing = config.framing ?? {};

  const updateConfig = (patch: Partial<RembrandtConfig>) => {
    onConfigChange({ ...config, ...patch });
  };

  const updateRender = (patch: Partial<RenderConfig>) => {
    updateConfig({ render: { ...render, ...patch } });
  };

  const updateOutput = (patch: Partial<OutputConfig>) => {
    updateConfig({ output: { ...output, ...patch } });
  };

  const updateBackground = (patch: Partial<BackgroundConfig>) => {
    updateConfig({ background: { ...background, ...patch } });
  };

  const updateLightRandomization = (patch: Partial<LightRandomizationConfig>) => {
    updateConfig({ light_randomization: { ...lightRandomization, ...patch } });
  };

  const updateLabels = (patch: Partial<LabelsConfig>) => {
    updateConfig({ labels: { ...labels, ...patch } });
  };

  const updateFraming = (patch: Partial<FramingConfig>) => {
    updateConfig({ framing: { ...framing, ...patch } });
  };

  const updateCamera = (patch: Partial<CameraConfig>) => {
    onCameraChange({ ...camera, ...patch });
  };

  const toggleLightType = (lightType: LightType, checked: boolean) => {
    const current = lightRandomization.light_types ?? [...LIGHT_TYPES];
    const next = checked
      ? [...new Set([...current, lightType])]
      : current.filter((value) => value !== lightType);
    updateLightRandomization({
      light_types: next.length > 0 ? next : [lightType],
    });
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
      <div className={styles.row}>
        <label className={styles.label} htmlFor="object-up-axis">
          Source up axis
        </label>
        <select
          id="object-up-axis"
          className={styles.select}
          value={config.object.up_axis ?? "Z"}
          onChange={(event) => onObjectUpAxisChange(event.target.value as SourceUpAxis)}
        >
          <option value="Y">Y-up OBJ</option>
          <option value="Z">Z-up OBJ</option>
        </select>
      </div>
      <label className={styles.checkboxRow}>
        <input
          type="checkbox"
          checked={config.object.normalize ?? true}
          onChange={(event) => onObjectNormalizeChange(event.target.checked)}
        />
        Normalize to unit size
      </label>
      <p className={styles.hint}>
        Distances and light energy assume a unit-sized object. Disable only if your config
        uses the asset&apos;s native units.
      </p>

      <h2 className={styles.sectionTitle}>Camera sampling</h2>
      <p className={styles.hint}>
        Framing jitter is applied at render time and is not shown in this preview.
      </p>
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

      <OptionalSeedInput
        id="camera-seed"
        label="Use camera seed"
        seed={camera.seed}
        onChange={(seed) => updateCamera({ seed })}
      />

      <h2 className={styles.sectionTitle}>Render</h2>
      <div className={styles.row}>
        <label className={styles.label} htmlFor="render-engine">
          Engine
        </label>
        <select
          id="render-engine"
          className={styles.select}
          value={render.engine ?? "EEVEE"}
          onChange={(event) =>
            updateRender({ engine: event.target.value as RenderConfig["engine"] })
          }
        >
          <option value="EEVEE">EEVEE</option>
          <option value="CYCLES">CYCLES</option>
        </select>
      </div>
      <div className={styles.row}>
        <label className={styles.label} htmlFor="focal-length">
          Focal length (mm)
        </label>
        <input
          id="focal-length"
          className={styles.input}
          type="number"
          min={1}
          max={300}
          step={1}
          value={render.focal_length ?? 50}
          onChange={(event) =>
            updateRender({
              focal_length: clampNumber(Number(event.target.value), 1, 300),
            })
          }
        />
      </div>
      <div className={styles.row}>
        <label className={styles.label}>Resolution (px)</label>
        <div className={styles.vec3Row}>
          <div className={styles.vec3Field}>
            <label className={styles.label} htmlFor="resolution-width">
              Width
            </label>
            <input
              id="resolution-width"
              className={styles.input}
              type="number"
              min={64}
              max={4096}
              step={1}
              value={render.resolution?.[0] ?? 640}
              onChange={(event) => {
                const width = Math.round(clampNumber(Number(event.target.value), 64, 4096));
                const height = render.resolution?.[1] ?? 640;
                updateRender({ resolution: [width, height] });
              }}
            />
          </div>
          <div className={styles.vec3Field}>
            <label className={styles.label} htmlFor="resolution-height">
              Height
            </label>
            <input
              id="resolution-height"
              className={styles.input}
              type="number"
              min={64}
              max={4096}
              step={1}
              value={render.resolution?.[1] ?? 640}
              onChange={(event) => {
                const height = Math.round(clampNumber(Number(event.target.value), 64, 4096));
                const width = render.resolution?.[0] ?? 640;
                updateRender({ resolution: [width, height] });
              }}
            />
          </div>
        </div>
      </div>
      <div className={styles.row}>
        <label className={styles.label} htmlFor="render-samples">
          Samples
        </label>
        <input
          id="render-samples"
          className={styles.input}
          type="number"
          min={1}
          max={4096}
          step={1}
          value={render.samples ?? 32}
          onChange={(event) =>
            updateRender({
              samples: Math.round(clampNumber(Number(event.target.value), 1, 4096)),
            })
          }
        />
      </div>

      <h2 className={styles.sectionTitle}>Background</h2>
      <div className={styles.row}>
        <label className={styles.label} htmlFor="background-mode">
          Mode
        </label>
        <select
          id="background-mode"
          className={styles.select}
          value={background.mode ?? "none"}
          onChange={(event) =>
            updateBackground({
              mode: event.target.value as BackgroundConfig["mode"],
            })
          }
        >
          <option value="none">none (flat color)</option>
          <option value="image">image (BG-20k)</option>
        </select>
      </div>
      {background.mode === "image" ? (
        <div className={styles.row}>
          <label className={styles.label} htmlFor="background-image-dir">
            Image directory
          </label>
          <input
            id="background-image-dir"
            className={styles.input}
            type="text"
            value={background.image_dir ?? ""}
            placeholder="backgrounds/bg20k"
            onChange={(event) =>
              updateBackground({
                image_dir: event.target.value.trim() === "" ? null : event.target.value,
              })
            }
          />
        </div>
      ) : null}
      <OptionalSeedInput
        id="background-seed"
        label="Use background seed"
        seed={background.seed}
        onChange={(seed) => updateBackground({ seed })}
      />
      <Vec3Input
        id="background-color"
        label="Flat background color (RGB 0–1)"
        value={background.color ?? [0.05, 0.05, 0.05]}
        onChange={(color) => updateBackground({ color })}
      />

      <h2 className={styles.sectionTitle}>Light randomization</h2>
      <div className={styles.row}>
        <label className={styles.label} htmlFor="light-randomization-mode">
          Mode
        </label>
        <select
          id="light-randomization-mode"
          className={styles.select}
          value={lightRandomization.mode ?? "static"}
          onChange={(event) =>
            updateLightRandomization({
              mode: event.target.value as LightRandomizationConfig["mode"],
            })
          }
        >
          <option value="static">static (use lights list)</option>
          <option value="random">random (per-frame rigs)</option>
        </select>
      </div>
      {lightRandomization.mode === "random" ? (
        <>
          <RangeInput
            id="light-count"
            label="Light count range"
            min={1}
            max={8}
            step={1}
            value={lightRandomization.count_range ?? [1, 3]}
            onChange={(count_range) => updateLightRandomization({ count_range })}
          />
          <fieldset className={styles.fieldset}>
            <legend className={styles.legend}>Light types</legend>
            {LIGHT_TYPES.map((lightType) => (
              <label key={lightType} className={styles.checkboxRow}>
                <input
                  type="checkbox"
                  checked={(lightRandomization.light_types ?? LIGHT_TYPES).includes(
                    lightType,
                  )}
                  onChange={(event) => toggleLightType(lightType, event.target.checked)}
                />
                {lightType}
              </label>
            ))}
          </fieldset>
          <RangeInput
            id="light-azimuth"
            label="Azimuth range (degrees)"
            min={0}
            max={360}
            step={1}
            value={lightRandomization.azimuth_range ?? [0, 360]}
            onChange={(azimuth_range) => updateLightRandomization({ azimuth_range })}
          />
          <RangeInput
            id="light-elevation"
            label="Elevation range (degrees)"
            min={-90}
            max={90}
            step={1}
            value={lightRandomization.elevation_range ?? [10, 80]}
            onChange={(elevation_range) => updateLightRandomization({ elevation_range })}
          />
          <RangeInput
            id="light-distance"
            label="Distance range"
            min={0.1}
            max={30}
            step={0.1}
            value={lightRandomization.distance_range ?? [4, 8]}
            onChange={(distance_range) => updateLightRandomization({ distance_range })}
          />
          <RangeInput
            id="light-energy-scale"
            label="Energy scale range"
            min={0.1}
            max={5}
            step={0.1}
            value={lightRandomization.energy_scale_range ?? [0.5, 2]}
            onChange={(energy_scale_range) =>
              updateLightRandomization({ energy_scale_range })
            }
          />
          <div className={styles.row}>
            <label className={styles.label} htmlFor="light-color-jitter">
              Color jitter (0–1)
            </label>
            <input
              id="light-color-jitter"
              className={styles.input}
              type="number"
              min={0}
              max={1}
              step={0.05}
              value={lightRandomization.color_jitter ?? 0}
              onChange={(event) =>
                updateLightRandomization({
                  color_jitter: clampNumber(Number(event.target.value), 0, 1),
                })
              }
            />
          </div>
          <RangeInput
            id="light-area-size"
            label="Area size range"
            min={0.1}
            max={10}
            step={0.1}
            value={lightRandomization.area_size_range ?? [1, 3]}
            onChange={(area_size_range) => updateLightRandomization({ area_size_range })}
          />
          <OptionalSeedInput
            id="light-randomization-seed"
            label="Use light randomization seed"
            seed={lightRandomization.seed}
            onChange={(seed) => updateLightRandomization({ seed })}
          />
        </>
      ) : null}

      <h2 className={styles.sectionTitle}>Static lights</h2>
      <StaticLightsPreview lights={config.lights ?? []} />

      <h2 className={styles.sectionTitle}>Framing</h2>
      <div className={styles.row}>
        <label className={styles.label} htmlFor="center-jitter">
          Center jitter (0 = centered)
        </label>
        <input
          id="center-jitter"
          className={styles.input}
          type="number"
          min={0}
          max={1}
          step={0.05}
          value={framing.center_jitter ?? 0.35}
          onChange={(event) =>
            updateFraming({
              center_jitter: clampNumber(Number(event.target.value), 0, 1),
            })
          }
        />
      </div>
      <RangeInput
        id="fill-range"
        label="Object fill range (fraction of frame height)"
        min={0.05}
        max={1}
        step={0.05}
        value={framing.fill_range ?? [0.15, 0.75]}
        onChange={(fill_range) => updateFraming({ fill_range })}
      />
      <OptionalSeedInput
        id="framing-seed"
        label="Use framing seed"
        seed={framing.seed}
        onChange={(seed) => updateFraming({ seed })}
      />

      <h2 className={styles.sectionTitle}>Labels</h2>
      <label className={styles.checkboxRow}>
        <input
          type="checkbox"
          checked={labels.enabled ?? true}
          onChange={(event) => updateLabels({ enabled: event.target.checked })}
        />
        Generate YOLO labels
      </label>
      <div className={styles.row}>
        <label className={styles.label} htmlFor="class-name">
          Class name
        </label>
        <input
          id="class-name"
          className={styles.input}
          type="text"
          value={config.object.class_name ?? "object"}
          onChange={(event) =>
            updateConfig({
              object: { ...config.object, class_name: event.target.value },
            })
          }
        />
      </div>
      <div className={styles.row}>
        <label className={styles.label} htmlFor="class-id">
          Class ID
        </label>
        <input
          id="class-id"
          className={styles.input}
          type="number"
          min={0}
          step={1}
          value={config.object.class_id ?? 0}
          onChange={(event) =>
            updateConfig({
              object: {
                ...config.object,
                class_id: Math.max(0, Math.round(Number(event.target.value))),
              },
            })
          }
        />
      </div>
      <div className={styles.row}>
        <label className={styles.label} htmlFor="min-visible-pixels">
          Min visible pixels
        </label>
        <input
          id="min-visible-pixels"
          className={styles.input}
          type="number"
          min={0}
          step={1}
          value={labels.min_visible_pixels ?? 25}
          onChange={(event) =>
            updateLabels({
              min_visible_pixels: Math.max(0, Math.round(Number(event.target.value))),
            })
          }
        />
      </div>

      <h2 className={styles.sectionTitle}>Output</h2>
      <div className={styles.row}>
        <label className={styles.label} htmlFor="output-dir">
          Output directory
        </label>
        <input
          id="output-dir"
          className={styles.input}
          type="text"
          value={output.dir ?? "output"}
          onChange={(event) => updateOutput({ dir: event.target.value })}
        />
      </div>
      <div className={styles.row}>
        <label className={styles.label} htmlFor="train-val-split">
          Train fraction (0–1)
        </label>
        <input
          id="train-val-split"
          className={styles.input}
          type="number"
          min={0.01}
          max={0.99}
          step={0.01}
          value={output.train_val_split ?? 0.8}
          onChange={(event) =>
            updateOutput({
              train_val_split: clampNumber(Number(event.target.value), 0.01, 0.99),
            })
          }
        />
      </div>
      <OptionalSeedInput
        id="split-seed"
        label="Use train/val split seed"
        seed={output.split_seed}
        onChange={(split_seed) => updateOutput({ split_seed })}
      />

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
