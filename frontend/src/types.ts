/** Shared types mirroring the FastAPI / pydantic API models. */

export type Vec3 = [number, number, number];
export type Vec2 = [number, number];
export type Bbox = [Vec3, Vec3];

export type SamplingStrategy = "random" | "fibonacci";
export type LightType = "POINT" | "SUN" | "AREA";
export type RenderEngine = "EEVEE" | "CYCLES";
export type BackgroundMode = "none" | "image";
export type LightRandomizationMode = "static" | "random";
export type BandEdgeKind = "azimuth" | "elevation";
export type SourceUpAxis = "Y" | "Z";

export type HealthResponse = {
  status: string;
};

export type PreviewMesh = {
  positions: number[];
  indices: number[];
  bbox: Bbox;
};

export type PreviewPosesParams = {
  bbox: Bbox;
  n: number;
  azimuth_range?: Vec2;
  elevation_range?: Vec2;
  distance_range?: Vec2;
  strategy?: SamplingStrategy;
  seed?: number | null;
  look_at?: Vec3;
};

export type BandShell = {
  distance: number;
  positions: number[];
  azimuth_count: number;
  elevation_count: number;
};

export type BandEdgeLine = {
  kind: BandEdgeKind;
  value_deg: number;
  positions: number[];
};

export type PreviewBand = {
  surface: BandShell;
  edges: BandEdgeLine[];
};

export type PreviewCameras = {
  locations: Vec3[];
  look_at: Vec3;
  rays: [Vec3, Vec3][];
};

export type PreviewGroundPlane = {
  positions: number[];
  indices: number[];
};

export type PreviewPoses = {
  band: PreviewBand;
  cameras: PreviewCameras;
  ground_plane: PreviewGroundPlane;
  display_radius: number;
};

export type ObjectConfig = {
  path: string;
  up_axis?: SourceUpAxis;
  class_name?: string;
  class_id?: number;
};

export type CameraConfig = {
  n: number;
  azimuth_range?: Vec2;
  elevation_range?: Vec2;
  distance_range?: Vec2;
  strategy?: SamplingStrategy;
  seed?: number | null;
  look_at?: Vec3;
};

export type LightConfig = {
  light_type?: LightType;
  location?: Vec3;
  look_at?: Vec3;
  energy?: number | null;
  color?: Vec3;
  size?: number;
};

export type RenderConfig = {
  focal_length?: number;
  resolution?: Vec2;
  engine?: RenderEngine;
  samples?: number;
};

export type OutputConfig = {
  dir?: string;
  train_val_split?: number;
  split_seed?: number | null;
};

export type BackgroundConfig = {
  mode?: BackgroundMode;
  image_dir?: string | null;
  seed?: number | null;
  color?: Vec3;
};

export type LabelsConfig = {
  enabled?: boolean;
  min_visible_pixels?: number;
};

export type FramingConfig = {
  center_jitter?: number;
  fill_range?: Vec2;
  seed?: number | null;
};

export type PostFxMode = "off" | "random";

export type PostFxConfig = {
  mode?: PostFxMode;
  gaussian_noise_sigma?: Vec2;
  blur_radius?: Vec2;
  jpeg_quality?: [number, number];
  exposure_ev?: Vec2;
  seed?: number | null;
};

export type LightRandomizationConfig = {
  mode?: LightRandomizationMode;
  count_range?: [number, number];
  light_types?: LightType[];
  azimuth_range?: Vec2;
  elevation_range?: Vec2;
  distance_range?: Vec2;
  energy_scale_range?: Vec2;
  color_jitter?: number;
  area_size_range?: Vec2;
  look_at?: Vec3;
  seed?: number | null;
};

export type RembrandtConfig = {
  object: ObjectConfig;
  camera: CameraConfig;
  lights?: LightConfig[];
  render?: RenderConfig;
  output?: OutputConfig;
  background?: BackgroundConfig;
  light_randomization?: LightRandomizationConfig;
  labels?: LabelsConfig;
  framing?: FramingConfig;
  postfx?: PostFxConfig;
};

export type SaveConfigResponse = {
  path: string;
};
