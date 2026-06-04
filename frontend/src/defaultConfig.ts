import type { RembrandtConfig } from "./types";

/** Defaults mirroring ``rembrandt.config.RembrandtConfig``. */
export function createDefaultConfig(objectPath = ""): RembrandtConfig {
  return {
    object: { path: objectPath, up_axis: "Z" },
    camera: {
      n: 10,
      azimuth_range: [0, 360],
      elevation_range: [-10, 30],
      distance_range: [3, 5],
      strategy: "random",
      seed: 42,
      look_at: [0, 0, 0],
    },
    lights: [
      {
        light_type: "SUN",
        location: [2, -3, 5],
        look_at: [0, 0, 0],
        energy: 3,
      },
      {
        light_type: "POINT",
        location: [-2, 2, 3],
      },
    ],
    render: {
      focal_length: 50,
      resolution: [640, 640],
      engine: "EEVEE",
      samples: 32,
    },
    output: {
      dir: "output",
      train_val_split: 0.8,
    },
    background: {
      mode: "none",
    },
  };
}
