# Rembrandt

Synthetic computer vision dataset tooling from 3D models, in honour of Rembrandt Harmenszoon van Rijn.

Rembrandt currently has two local tools:

- `rembrandt-serve`: a browser configurator for choosing an `.obj`, previewing camera-angle coverage in 3D, and saving a YAML config.
- `rembrandt-render CONFIG_PATH`: a Blender / `bpy` render step that reads that YAML and writes a YOLO training dataset (images, labels, `data.yaml`, and training handoff files).

The SPA preview is a configuration-sanity tool. It shows the sampled camera band relative to the oriented object and ground plane so you can decide whether azimuth/elevation coverage makes sense before running Blender. It does not preview rendered images, trigger renders, or monitor render progress.

## Setup

### 1. System libraries (for Blender / bpy)

On Debian/Ubuntu:

```bash
sudo apt install libxrender1 libxi6 libxxf86vm1 libxfixes3 libxkbcommon0 libsm6 libgl1
```

### 2. Python 3.11 backend

Use Python 3.11. The `bpy` wheel is tied to the Python minor version, so other Python versions are not supported.

```bash
pip install -e ".[dev]"
```

### 3. Frontend dependencies and build

The frontend uses Yarn. If `yarn` is not already available, enable it through Corepack first:

```bash
corepack enable
```

Then build the SPA:

```bash
cd frontend && yarn install && yarn build
```

Or from the repo root: `make build-frontend`

## Quick Start

### Configure

```bash
rembrandt-serve
```

This starts the FastAPI server at [http://127.0.0.1:8000/](http://127.0.0.1:8000/) and opens the browser. Paste an `.obj` path, tune the camera controls, then save a config. Saved configs are written under `./configs/` relative to the server working directory.

### Render

```bash
rembrandt-render ./configs/dataset.yaml
cd output/<stamp>/dataset && pip install ultralytics && python train_yolo.py
```

By default, output goes to `<output.dir>/<timestamp>/dataset/` with `images/{train,val}`,
`labels/{train,val}`, `data.yaml`, `train_yolo.py`, and `README.md`. Use `--frames-only` to
write flat `frame_*.png` files under `<output.dir>/<timestamp>/` instead.

Each run also writes `<output.dir>/<timestamp>/run.json` with the resolved config, object
path, and per-frame camera/light/background metadata for debugging.

`output.dir` may be absolute, relative to the config file, or relative to the current working
directory. Resolution order: absolute → as-is; else config-relative; else CWD if it exists
there; for a new directory, created next to the config. Object paths follow a similar order
(absolute → config-relative → CWD).

The default render engine is `EEVEE`, which requires a GPU when running headless `bpy`. On
CPU-only machines (typical CI runners and many servers), use `render.engine: CYCLES` instead.
Rembrandt raises an explicit error rather than silently switching engines mid-dataset.

Use `--workers N` to render frames in parallel across `N` separate Blender processes (one
frame subset per worker). Pose, lighting, background, framing, and post-fx sampling are
deterministic per frame index, so parallel runs produce the same dataset as a single process.

Use `--stats` after a run to print label class distributions.

## Config Format

The SPA writes the same YAML schema that `rembrandt-render` consumes:

```yaml
object:
  path: /absolute/path/to/model.obj
  # Native OBJ up-axis. Defaults to Z; set Y for legacy Y-up models.
  up_axis: Z
  class_name: my_object
  class_id: 0
camera:
  n: 10
  azimuth_range: [0.0, 360.0]
  elevation_range: [-10.0, 30.0]
  distance_range: [3.0, 5.0]
  strategy: random
  seed: 42
  look_at: [0.0, 0.0, 0.0]
lights:
  - light_type: SUN
    location: [2.0, -3.0, 5.0]
    look_at: [0.0, 0.0, 0.0]
    energy: 3.0
  - light_type: POINT
    location: [-2.0, 2.0, 3.0]
render:
  focal_length: 50.0
  resolution: [640, 640]
  engine: EEVEE
  samples: 32
output:
  dir: output
  train_val_split: 0.8
  split_seed: null
labels:
  enabled: true
  min_visible_pixels: 25
framing:
  center_jitter: 0.35
  fill_range: [0.15, 0.75]
# Optional: randomized photo backgrounds (default mode is none)
# background:
#   mode: image
#   image_dir: ./backgrounds
#   seed: 7
# Optional: per-frame randomized light rigs (default mode is static)
# light_randomization:
#   mode: random
#   count_range: [1, 3]
#   light_types: [POINT, SUN, AREA]
#   seed: 7
```

`output.train_val_split` controls the train/val partition written by the dataset layout.
Set `output.split_seed` for a reproducible split (`null` means non-reproducible). This seed
is independent of `camera.seed` and other config seeds.

### Labels

YOLO bounding boxes are derived from the rendered alpha mask (not projected geometry at
runtime). `labels.enabled` defaults to `true`. Frames with fewer than
`labels.min_visible_pixels` foreground pixels get an empty label file (negative examples).

### Framing

When framing is active, `camera.distance_range` is a lower bound on camera distance; sampled
`framing.fill_range` controls how large the object appears. `framing.center_jitter` offsets the
look-at point in the image plane for in-frame translation diversity. Set `framing.seed` for
a reproducible sequence (`null` means non-reproducible).

### Post-fx

`sensor-domain` effects (`postfx.mode: random`) simulate camera artifacts after label
extraction. All post-fx is geometry-preserving — anything that moves pixels belongs in 3D,
not here. Set `postfx.seed` for reproducibility.

### Randomized lighting

When `light_randomization.mode` is `random`, Rembrandt builds a fresh light rig before each frame: randomized count, types (POINT / SUN / AREA), positions on a spherical band around `look_at` (same +Z-up convention as the camera sampler), energy scaled relative to per-type defaults, optional color jitter, and AREA size. Set `light_randomization.seed` for a reproducible rig sequence (`null` means non-reproducible). This seed is independent of `camera.seed` and `background.seed`.

In `random` mode the static `lights:` list is **ignored** (not merged). `energy_scale_range` multiplies type defaults from `light_poses.DEFAULT_LIGHT_ENERGY` — 1000 W for POINT, 5 for SUN, 100 W for AREA — because absolute energy values are not comparable across types. For SUN lights only the direction from `location` toward `look_at` affects shading.

The default elevation band `(10, 80)` keeps sampled lights above the ground plane; widen `elevation_range` explicitly if you need light from below.

### Randomized backgrounds

When `background.mode` is `image`, Rembrandt renders each frame with a transparent film (RGBA), then alpha-composites the object over a randomly chosen photo from `background.image_dir`. The foreground pixels are never moved or scaled — only the background is resized/cropped to cover the frame. Set `background.seed` for a reproducible background sequence per run (`null` means non-reproducible). This seed is independent of `camera.seed`.

With `background.mode: none` (the default), frames are composited over `background.color`
instead of the Blender world background. Cast shadows are not preserved in composited mode
(no shadow catcher yet) — expect a floating-object look until that lands.

Recommended workflow using [BG-20k](https://huggingface.co/datasets/unography/BG-20k-1200px) (20k high-resolution photos without salient objects, MIT license):

```bash
pip install -e ".[backgrounds]"
rembrandt-fetch-backgrounds --out ./backgrounds --count 2000
# then in your config:
#   background:
#     mode: image
#     image_dir: ./backgrounds
#     seed: 7
```

BG-20k was built for compositing: backgrounds contain no salient objects, so rendered object classes cannot appear unlabeled in a background once YOLO labels exist. If you redistribute a dataset built with these backgrounds, cite *Bridging Composite and Real: Towards End-to-End Deep Image Matting* (IJCV 2021).

Any local directory of `.jpg`/`.jpeg`/`.png`/`.webp` images works — the fetch command is optional. If you use **Open Images** instead, download a bounded subset (for example via FiftyOne: `foz.load_zoo_dataset("open-images-v7", split="validation", max_samples=N)`) — never the full set. Filter backgrounds by class labels if your rendered object class might appear in a photo (label leakage). Open Images annotations are CC-BY; photos are Flickr images under CC-BY-style licenses — attribution applies on redistribution.

## Development

Run the API and Vite dev server together:

```bash
make dev
```

- API: [http://127.0.0.1:8000](http://127.0.0.1:8000) (uvicorn with reload)
- Frontend: [http://127.0.0.1:5173](http://127.0.0.1:5173) (proxies `/api` to the API)

Useful checks:

```bash
pytest -q
ruff check src tests
ruff format --check src tests
mypy --no-sqlite-cache src
cd frontend && yarn typecheck && yarn build
```

### Tests (two lanes)

Most tests are **bpy-free** and run with the dev install above:

```bash
pytest -m "not bpy" -q
```

Orientation parity (`orient_and_center` vs `bpy.ops.wm.obj_import`, preview mesh vs
scene geometry) lives in bpy-marked tests. Run them locally when Blender is installed:

```bash
pytest -m bpy -q
```

CI runs both lanes; the bpy job uses `pytest -m bpy --require-bpy` so a missing Blender
runtime fails loudly instead of skipping. Optional full-size sample assets live under
`test-obj/` (gitignored); committed fixtures under `tests/fixtures/` cover parity in CI.

To reproduce the original orientation report locally, copy
`12951_Stone_Chess_Board_v1_L3.obj` into `test-obj/` and run:

```bash
pytest tests/test_convention.py::test_orient_and_center_matches_bpy_import_on_chess_board_object \
  tests/test_render_orientation.py::test_rendered_view_keeps_world_z_upright_on_chess_board_object \
  -v --require-bpy
```

The old Streamlit / Plotly camera-pose preview has been retired. The React + Three.js SPA is now the only preview UI.

Copyright @conchiglia 2026
