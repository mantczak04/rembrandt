# Rembrandt — From Frame Renderer to Dataset Generator: Gap Analysis & Implementation Plan

> Scope: close the gap between the current state (config → PNG frames) and the product goal
> (config → ready-to-train YOLO dataset + training handoff). Written against commit state of
> the uploaded snapshot (June 2026). Follows the repo's hard rules from `.ai/AGENTS.md`:
> bpy only in `scene.py` / lazy in `camera/orientation.py`; new logic in pure, bpy-free
> modules with bpy-free tests; Python is the source of truth, frontend only displays.

---

## 1. Current state (what already works)

| Goal step | Status |
|---|---|
| 1. Pick a 3D model (.obj) | ✅ Done — path-based, `up_axis` Y/Z convention, MTL/texture repair (`obj_assets.py`) |
| 2. Config: count, angles, lighting, backgrounds | ✅ Mostly — camera band sampling (random/fibonacci, sin-uniform elevation), static + randomized light rigs, post-composite photo backgrounds (BG-20k fetcher). ❌ No noise/post-fx, ❌ no masking output, ❌ UI exposes only object + camera |
| 3. Images generated from config | ✅ Done — `rembrandt-render`, EEVEE/CYCLES, camera fit, transparent-film compositing |
| 4. YOLO bounding boxes | ❌ **Not implemented at all** — no annotation module, no labels, no dataset layout, `train_val_split` is parsed but ignored |
| 5. Copy-paste training code | ❌ Not implemented |

## 2. Gaps and issues found

### Blocking the product goal

**G1 — No labels.** Nothing computes 2D bounding boxes. This is the core missing feature.

**G2 — No dataset layout.** Frames land in `output/<timestamp>/frame_XXXX.png`. Ultralytics
YOLO expects `images/{train,val}`, `labels/{train,val}`, and a `data.yaml`. `output.train_val_split`
exists in the schema but is dead config.

**G3 — No class identity.** The config has no concept of a class name/id. A YOLO label line
needs a class index; `data.yaml` needs `names`.

**G4 — Every image is a centered, fully-framed object.** `center_target()` puts the bbox
center at the origin, every camera `look_at`s the origin, and `fit_target=True` pushes the
camera back so the object always fits with the same margin. Result: the object is always
near image center at near-constant relative scale. A detector trained on this generalizes
poorly (it learns "the object is always in the middle"). Translation/scale diversity is a
known major factor for synthetic-data detection quality. This is a *data-quality* gap, not
a code bug — but it will hurt step 5's outcome more than anything except G1.

### Correctness bugs / risks

**B1 — Multi-mesh OBJ inconsistency.** `Scene.load_object()` keeps only the *first* mesh
(`imported[0]`), and `center_target()` / `_fit_camera_to_target()` use only `self.target`'s
`bound_box`. But `preview/mesh.py` parses *all* vertices in the file. For a multi-mesh OBJ:
preview centers/frames the union; render centers/frames only mesh #1, and any future bbox
projection would label only mesh #1. The comment "take the first for now" acknowledges it,
but once labels exist this becomes silent label corruption, so it should be fixed in Stage 0.

**B2 — EEVEE in headless bpy needs a GPU.** `BLENDER_EEVEE_NEXT` in the bpy 4.5 wheel
requires a working GPU/EGL context. On a CPU-only box (typical CI runner, many servers) the
render either fails or produces garbage. There's no guardrail or documented fallback. Default
engine in the schema is EEVEE.

**B3 — `output.dir` resolution is inconsistent with the rest of the config.** Object path and
`background.image_dir` resolve relative to the config file first; `output.dir` resolves
relative to CWD only (`Path(cfg.output.dir) / run_stamp`). Surprising when invoking from
another directory.

**B4 — Shadows disappear in background mode.** With `film_transparent=True` the object's cast
shadow is not in the alpha, so composited images show a "floating" object. Acceptable for v1
(BG-20k images aren't ground planes anyway), but worth a config-documented note and a future
shadow-catcher option, since "objects floating with no contact shadow" is a known sim-to-real
gap.

### Product / UX gaps

**U1 — The SPA can only edit object + camera.** `SaveBar` writes `createDefaultConfig()`
merged with camera params — lights, render settings, background, and light randomization are
silently saved with defaults the user never saw. Today users must hand-edit YAML for the
features that matter most for dataset realism.

**U2 — No single "generate dataset" command.** The end state of the user journey should be
one command from a config to a trainable folder.

**U3 — No per-frame metadata.** Debugging a bad dataset ("which pose/lights/background made
this frame?") currently requires reading stdout. A sidecar JSON per run is cheap.

### Deliberately out of scope (and why)

- **Heavy 2D augmentation (mosaic, flips, HSV...)** — Ultralytics does this at train time,
  better and on-the-fly. Rembrandt should only add *sensor-domain* effects YOLO doesn't
  simulate (noise, blur, JPEG artifacts) — Stage 3.
- **Triggering renders from the SPA** — violates the AGENTS.md hard rule (configurator is a
  preview-sanity tool). Revisit only as an explicit, separate decision.
- **Formats beyond YOLO** — design the annotation core so COCO export is a writer away
  (Stage 7), but don't build it now.

---

## 3. Plan

Stages ordered by value-per-effort: Stage 1 alone makes Rembrandt usable end-to-end;
Stage 2 makes the resulting models actually good.

---

### Stage 0 — Correctness groundwork (small, do first)

**0.1 Multi-mesh support in `Scene`.**
Replace `self.target: Object | None` with `self.targets: list[Object]` (keep a
`self.target` property returning `targets[0]` for compatibility during the change, then
remove it). Update:
- `center_target()` — compute the union AABB over `obj.matrix_world @ Vector(corner)` for
  every corner of every imported mesh; translate *all* meshes by `-center`.
- `_fit_camera_to_target()` — `radius = max over all meshes' corners of (corner − look_at).length`.
Do **not** use `bpy.ops.object.join()` — it mutates user data semantics (materials merge,
object names lost) and needs fragile active-object context. Iterating the list is simpler
and side-effect-free.
Add a bpy-marked test: two-cube OBJ fixture (offset cubes), assert union bbox center lands
at origin and preview-mesh bbox (which already unions everything) matches scene union bbox —
this extends the existing `test_orientation_parity.py` pattern.

**0.2 Engine guardrail.**
In `Scene.render`, when `engine == "EEVEE"`, wrap the render in a try and detect GPU-context
failure; raise a custom `RenderEngineUnavailableError` (add to `errors.py`) with the message
"EEVEE requires a GPU in headless bpy; use engine: CYCLES or run on a machine with a GPU."
Document in README. (Don't silently fall back — silent engine switches change image
statistics mid-dataset.)

**0.3 `output.dir` resolution.**
Add `resolve_output_dir(config_path, dir)` in `render.py` mirroring `resolve_object_path`
(absolute → as-is; else relative to config file; else CWD). Pure function + tests in the
bpy-free lane. Document in README config section.

**0.4 Run metadata sidecar.**
`render_from_config` writes `<run_dir>/run.json`: full resolved config (`model_dump`),
resolved object path, per-frame records `{frame, camera_pose, light_rig?, background?}`.
Pure-Python; populate the records inside the existing loop. This becomes load-bearing in
Stage 1 (the dataset writer reads it) and Stage 2 (debugging framing).

---

### Stage 1 — YOLO labels + dataset output (the MVP)

Design decision up front: **derive bboxes from the rendered alpha channel (object mask),
not from projected vertices**, with vertex projection kept as a parity *test*.

Why alpha-mask as primary:
- The compositing pipeline already renders RGBA with `film_transparent=True` when backgrounds
  are on. Making transparent-film the *always-on* render mode (compositing over background
  photo *or* a flat/world color) means the mask is free.
- The mask is exactly the rendered silhouette: it automatically accounts for what's actually
  visible in the image, anti-aliasing, and any future truncation from Stage 2 framing jitter.
  Projected-vertex boxes are amodal and go wrong the moment the object can leave the frame.
- It gives YOLO-seg polygons and instance masks for free later (the user explicitly wants
  masking as an option).
Cost: one extra in-memory image read per frame — negligible next to render time.

**1.1 Pure module `src/rembrandt/annotations.py` (NO bpy).**
Functions:
- `mask_from_alpha(rgba: NDArray[u8], *, threshold: int = 8) -> NDArray[bool]` —
  `rgba[..., 3] >= threshold`. Threshold 8 (not 1) skips faint AA fringe pixels.
- `bbox_from_mask(mask) -> tuple[int, int, int, int] | None` — pixel-space inclusive
  `(x0, y0, x1, y1)` via `np.flatnonzero(mask.any(axis=0))` / `axis=1`; `None` if empty.
- `yolo_line(class_id: int, bbox_px, *, width: int, height: int) -> str` — normalized
  `cx cy w h`: `cx = (x0 + x1 + 1) / 2 / width`, `w = (x1 - x0 + 1) / width`, formatted
  `f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"`.
- `visible_pixel_count(mask) -> int`.
Full bpy-free test coverage: synthetic masks (single blob, empty, 1-px, touching-edge),
round-trip normalization, threshold behavior.

**1.2 Config: class + labeling section.**
- `ObjectConfig`: add `class_name: str = "object"` and `class_id: int = Field(default=0, ge=0)`.
- New `LabelsConfig`: `enabled: bool = True`, `min_visible_pixels: int = Field(default=25, ge=0)`.
  Below the visibility floor, write an **empty label file** (Ultralytics treats those as
  background/negative images — that's desirable, not an error) but log it.
- Mirror in `frontend/src/types.ts` + `defaultConfig.ts` (display only; no logic — hard rule 2).

**1.3 Render loop: always-transparent film + label emission.**
In `render_from_config`:
- Always render with `transparent_film=True` when `labels.enabled` (i.e., effectively always).
- After `scene.render(...)`, load the RGBA once with PIL→numpy; compute mask → bbox → label
  *before* compositing, then composite:
  - background mode `image`: existing `composite_over` path (refactor
    `apply_background_to_frame` to accept the already-loaded foreground array to avoid a
    second disk read).
  - background mode `none`: composite over a flat color (new `BackgroundConfig.color:
    tuple[float,float,float] = (0.05, 0.05, 0.05)` — matches today's dark world look) so
    output stays RGB and visually unchanged for current users.
- Write `frame_XXXX.txt` next to the PNG (flat run dir; the dataset writer reorganizes).
Keep `render.py` bpy-free-by-delegation as it is now (numpy/PIL are fine there — same as the
existing backgrounds usage).

**1.4 Pure module `src/rembrandt/dataset.py` (NO bpy) — the dataset writer.**
- `split_indices(n: int, train_fraction: float, *, seed: int | None) -> tuple[list[int], list[int]]` —
  `Random(seed).shuffle` of `range(n)`, `n_val = max(1, round(n * (1 - train_fraction)))` when
  `n >= 2`, else everything in train. Deterministic, tested for edge cases (n=1, n=2,
  fraction extremes).
- `write_yolo_dataset(run_dir, out_dir, *, class_names: dict[int, str], split, seed)` —
  moves/copies frames+labels into
  `out_dir/{images,labels}/{train,val}/` and writes `data.yaml`:
  ```yaml
  path: .            # relative; Ultralytics resolves against the yaml's own dir
  train: images/train
  val: images/val
  names:
    0: <class_name>
  ```
  Use `shutil.move` within the run dir (rename is atomic on same fs) rather than copy —
  datasets get large.
- `OutputConfig`: add `split_seed: int | None = None` (independent of camera/light/bg seeds,
  consistent with the project's seed-independence convention).

**1.5 Wire into the CLI.**
Extend `rembrandt-render` to run the dataset writer after the frame loop when
`labels.enabled`, producing `<output.dir>/<stamp>/dataset/` with the YOLO layout, and print
the `data.yaml` path as the final line. (No new command needed — "render" now means "render
a dataset". Keep a `--frames-only` flag to skip labeling/layout for debugging.)

**1.6 Parity test (bpy lane): projected-vertex bbox vs mask bbox.**
Using the existing chess/cube fixtures: project all evaluated-mesh world vertices through
`K @ [R|t]` — `K` from the already-ported `intrinsics_as_k_matrix(...)`, extrinsics from
`camera.matrix_world.inverted()` with the Blender cam convention (look −Z, up +Y: flip Y/Z
rows, i.e. multiply rows 1,2 of the camera-space coords by −1 before applying K). Assert the
mask bbox ⊆ projected bbox and IoU > 0.9 for a fully-visible object. This pins the labeling
math to ground truth without making vertex projection a runtime dependency.

**Stage 1 exit criteria:** `rembrandt-render config.yaml` →
`output/<stamp>/dataset/{images,labels}/{train,val} + data.yaml`, and
`yolo detect train data=.../data.yaml model=yolo11n.pt` runs without dataset errors.

---

### Stage 2 — Framing diversity (make the datasets actually train well)

Goal: break "object always centered, always same relative size" (G4) while keeping
perspective geometrically correct.

**2.1 `FramingConfig` (new config section, advisory defaults on).**
```yaml
framing:
  center_jitter: 0.35      # fraction of the half-frame the bbox center may wander from image center
  fill_range: [0.15, 0.75] # target fraction of image height the object's projected height may occupy
  seed: null               # independent, consistent with other seeds
```

**2.2 Implementation — do it in 3D, not by pasting pixels.**
Two mechanisms, both pure math in a new bpy-free `framing.py`, consumed by the render loop:
- **Scale diversity:** today `fit_target=True` clamps distance to "object fills frame with
  margin 1.2". Replace the per-frame margin with a sampled one: per frame draw
  `fill ∈ fill_range`, convert to a fit margin `margin = 1 / fill`, and pass it through
  `move_camera(fit_margin=...)` with `fit_target=True`. The existing fit math
  (`fit_distance`, `limiting_fov_from_camera`) already does the hard part; distance_range
  becomes a *lower bound* input rather than the visible-size determinant. Document this
  clearly — it changes the meaning of `distance_range` when framing is enabled.
- **Translation diversity:** jitter `look_at` per frame. Sample an offset in the camera's
  image plane (perpendicular to the view direction — this is the correct way to move the
  object in-frame without changing its appearance): build the camera basis (right, up) from
  the view direction with world +Z reference, offset
  `look_at' = look_at + (u·right + v·up) * d * tan(fov/2) * center_jitter`, `u,v ∈ U(-1,1)`,
  where `d` is the fitted camera distance. This puts the object center anywhere within
  ±`center_jitter` of frame center.
- The object may now be partially out of frame at high jitter — that's *good* (truncation
  robustness), and the Stage 1 mask-based labels handle it correctly by construction
  (clamped to visible pixels; empty-label fallback below `min_visible_pixels`).

**2.3 Preview honesty.**
Per AGENTS.md hard rule 5 the preview is about angles, not distance — keep it that way, but
add a one-line note in the SPA ("framing jitter is applied at render time and not shown
here") once U1 lands. No new preview geometry needed.

**2.4 Validation step.**
Add a tiny analysis helper (`rembrandt-render ... --stats` or a notebook-free
`dataset.py:summarize_labels`) that prints the distribution of bbox centers and heights from
the generated labels. Acceptance: centers spread across the configured jitter region; heights
spanning `fill_range`. This is the cheapest way to *see* G4 is fixed.

---

### Stage 3 — Sensor-domain post-fx (optional realism knobs)

Only effects YOLO's own train-time augmentation does **not** simulate; all post-composite,
all pure PIL/numpy, new bpy-free module `postfx.py`, applied after compositing and **before**
label writing is unnecessary (none of these move pixels — geometry-preserving only, so labels
stay valid; enforce that rule in the module docstring).

```yaml
postfx:
  mode: random            # off by default
  gaussian_noise_sigma: [0.0, 8.0]   # in 8-bit units
  blur_radius: [0.0, 1.2]            # PIL GaussianBlur
  jpeg_quality: [55, 95]             # encode->decode round trip via BytesIO
  exposure_ev: [-0.7, 0.7]           # multiply by 2**ev in linear-ish space
  seed: null
```
Each effect independently sampled per frame; record sampled values in `run.json`. Tests:
determinism per seed, shape/dtype preservation, no-op when mode off.

---

### Stage 4 — Training handoff

**4.1 Generate the snippet (yes — but as files, not stdout).**
Into `dataset/`: a short `README.md` and `train_yolo.py`:
```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
model.train(data="data.yaml", epochs=100, imgsz=<render width>, batch=16)
metrics = model.val()
```
`imgsz` taken from `render.resolution`. README covers: `pip install ultralytics`, the one
command, a note that Ultralytics applies its own augmentations (so Rembrandt's postfx should
not try to replicate them), and the BG-20k citation line when `background.mode: image`
(carry it through from the fetcher — license hygiene the README already cares about).
Generating a snippet is a good idea precisely because it's static text — no execution, no
dependency on ultralytics inside Rembrandt.

**4.2 Smoke-check the contract in CI (cheap version).**
Don't install ultralytics in CI (heavy). Instead, validate `data.yaml` structure and
label-file grammar with a pure test (`every line: int + 4 floats in [0,1]`, files pair with
images 1:1). That covers 95% of "dataset won't load" failures.

---

### Stage 5 — SPA parity (close U1)

Order: only after Stage 1–2, because the UI should expose the *final* schema once, not chase it.

**5.1** Controls sections for: render (engine, resolution, samples, focal length), background
(mode, dir, seed, color), light randomization (mode + ranges), framing, labels (class name),
output (dir, split, seeds). All plain form state → saved config; zero computation in TS
(hard rule 2). Follow `ct-frontend-design` patterns as `.ai/AGENTS.md` already mandates.
**5.2** Surface what will actually be saved: a read-only YAML preview pane (serialize the
config object client-side for display only) so "silently saved defaults" can't happen again.
**5.3** Static lights remain YAML-only for now (positioning lights in a form is poor UX;
randomized rigs are the recommended path anyway) — just render the static list read-only.

---

### Stage 6 — Throughput ("seconds, not hours")

**6.1 Parallel rendering via OS processes.**
bpy is a process-global singleton — `ProcessPoolExecutor`/threads inside one interpreter
won't work. Mechanism: add `rembrandt-render CONFIG --frame-range A:B --run-dir DIR`
(internal flags); a coordinator (`--workers N`) precomputes nothing extra — pose/light/bg
sampling is already deterministic per `frame_index` and seeds, so worker `k` of `N` renders
indices `k, k+N, k+2N, ...` into the *same* run dir with the same stamp, then the coordinator
runs the dataset writer once at the end. Each worker = `subprocess.run([sys.executable, "-m",
"rembrandt.render", ...])`. Caveat to verify first: `light_poses.sample_light_rig` /
`choose_background` must depend only on `(seed, frame_index)` — they already do; add a test
asserting frame-level determinism so parallelism can never change outputs.
**6.2 Scene reuse is already done** (one import, camera moved per frame) — the remaining
per-frame cost is the render itself, so workers are the only real lever besides GPU EEVEE.

---

### Stage 7 — Later / optional backlog (not scheduled)

- **YOLO-seg export:** masks already exist; add polygonization (OpenCV `findContours` +
  `approxPolyDP`, or pure-python marching squares via scikit-image) → `class x1 y1 x2 y2 ...`.
- **COCO JSON writer** alongside YOLO from the same per-frame mask/bbox records.
- **Distractor objects** (random primitives with random materials in-scene) — the single
  biggest remaining realism lever for detection after backgrounds + framing.
- **Material/texture randomization** of the target.
- **Shadow catcher** option for background mode (Cycles `is_shadow_catcher` plane → keeps
  contact shadows in the alpha composite), addressing B4.
- **Multi-object / multi-class scenes** — requires per-object masks (Cycles object-index
  pass / cryptomatte), a real schema change; explicitly out of scope until single-class is
  proven.

---

## 4. Suggested module map after Stage 1–3

```
src/rembrandt/
├── annotations.py     # mask→bbox→YOLO line. NO bpy.        (Stage 1)
├── dataset.py         # split, layout, data.yaml, stats. NO bpy. (Stage 1, 2.4)
├── framing.py         # fill/jitter sampling + image-plane offset math. NO bpy. (Stage 2)
├── postfx.py          # noise/blur/jpeg/exposure. NO bpy.   (Stage 3)
└── (existing modules unchanged in role)
```

## 5. Sequencing summary

| Stage | Outcome | Size |
|---|---|---|
| 0 | Multi-mesh correctness, engine guardrail, path/metadata hygiene | S |
| 1 | **End-to-end: config → trainable YOLO dataset** | M–L |
| 2 | Framing diversity → models that generalize | M |
| 3 | Sensor post-fx | S |
| 4 | data.yaml + train_yolo.py handoff | S |
| 5 | SPA exposes full schema | M |
| 6 | Parallel workers | M |