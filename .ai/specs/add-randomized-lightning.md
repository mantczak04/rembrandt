# Rembrandt — Randomized Lighting (Per-Frame Light Rigs): Implementation Plan

> **For the implementing agent (Composer, Cursor):** Read this whole file first, then
> `.ai/AGENTS.md`, `src/rembrandt/camera_poses.py` (the style and seeding
> reference this plan deliberately mirrors), `src/rembrandt/scene.py`
> (`add_light`, the energy-defaults dict, `clear`), `src/rembrandt/config.py`,
> and `src/rembrandt/render.py`. Execute the tasks **in order** — each lists its
> dependencies, files, steps, and acceptance criteria. Do not start a task until
> its dependencies pass their acceptance gates.

---

## 1. Overview

### What we're building

Today every frame in a render run shares one static light list from the config
(`lights:`). For detection-robustness domain randomization we want **a fresh
light rig per frame**: a randomized number of lights, randomized **types**
(POINT / SUN / AREA), randomized **angles and distances** on a spherical band
around the object (the same spherical convention the camera sampler uses),
randomized **energy** (scaled relative to each type's default, since the types
use different units), optional **color jitter**, and randomized **AREA size**.

The shape of the solution mirrors the camera sampler exactly:

1. A new **pure, bpy-free** module `src/rembrandt/light_poses.py` samples
   per-frame light rigs deterministically (`Random(seed + frame_index)`),
   reusing the spherical math already in `camera_poses.py`.
2. A new `light_randomization` config block, default `mode: "static"` so all
   existing configs behave exactly as today.
3. `Scene` gains one small bpy addition — `clear_lights()` — so the render
   loop can swap rigs between frames without rebuilding the whole scene.
4. `render_from_config` wires it together: in `random` mode it clears and
   rebuilds the light rig before each frame; in `static` mode the existing
   once-before-the-loop behavior is untouched.

### Why per-frame rigs sampled in pure Python

- Lighting variation is one of the highest-value domain-randomization axes for
  synthetic detection data — fixed lighting bakes a single shading pattern
  into every image of the dataset.
- A light placement is just a `(location, look_at)` pair — structurally
  identical to a camera pose — so the sampler reuses the proven, tested
  spherical-band math instead of duplicating it (Guardrail: one spherical
  convention, defined once).
- Keeping the sampler pure keeps it in the fast test lane and keeps Python the
  single source of truth, per `.ai/AGENTS.md`.

### Out of scope (do **not** build here)

- HDRI / environment / image-based lighting, light linking, gobos, volumetrics.
- Randomizing the light **target**: all sampled lights aim at a fixed
  `look_at` (default the origin, where the object is centered). No look-at
  jitter in v1.
- Per-frame randomization of anything other than lights (no material, camera
  focal-length, or object-pose randomization — separate roadmap items).
- SPA UI or preview geometry for lights. The preview is about camera-angle
  coverage; lights have no preview meaning. (A minimal type mirror in the
  frontend is included in T5 only so `createDefaultConfig` stays an honest
  mirror of `RembrandtConfig`.)
- Removing or changing the static `lights:` list — it remains the `static`
  mode and the default.

---

## 2. Guardrails (apply to every task)

1. **The bpy-free boundary is unchanged.** `light_poses.py` and `config.py`
   must not `import bpy` or import any module that does. The only bpy change
   in this plan is `Scene.clear_lights()` in `scene.py`. `render.py` must keep
   passing `test_render_module_only_imports_bpy_through_scene`. Add a
   `test_light_poses_module_is_bpy_free` source-scan test mirroring the
   existing pattern.
2. **One spherical convention, defined once.** Light positions use the same
   +Z-up, elevation-from-XY-plane, sin-elevation-area-uniform sampling as the
   camera. Do **not** re-derive the spherical→Cartesian math in
   `light_poses.py` — reuse the helper promoted from `camera_poses.py` (T1).
   Same for range validation (T2).
3. **One source of truth for per-type energy defaults.** The
   `{"POINT": 1000.0, "SUN": 5.0, "AREA": 100.0}` dict currently inlined in
   `Scene.add_light` moves to the pure layer (T1) and `scene.py` imports it
   from there. The sampler's `energy_scale_range` multiplies these defaults —
   never absolute energy ranges, because the three types use different units
   (POINT/AREA in Watts, SUN unitless) and one absolute range cannot be valid
   across types.
4. **Determinism follows the existing seeding discipline.** One local
   `random.Random(seed + frame_index)` per frame drives **everything** in that
   frame's rig (count, types, positions, energies, colors, sizes), exactly
   like `choose_background`. Same config → byte-identical rig sequence. Never
   mutate global RNG state (replicate the
   `test_sample_camera_poses_does_not_mutate_global_rng` pattern).
5. **Existing behavior is the default.** `light_randomization.mode` defaults
   to `"static"`; every existing YAML config, SPA-saved config, and current
   test must load and behave unchanged. In `static` mode, `clear_lights()` is
   never called and lights are added once before the loop, as today.
6. **No light-data leaks across frames.** `clear_lights()` must remove both
   the light **objects** and their **data blocks** (`bpy.data.lights`) —
   otherwise thousands of frames accumulate orphaned data blocks. A bpy-marked
   test asserts both collections are clean after clearing.
7. **Sensible defaults light from above the ground plane.** Default
   `elevation_range` is `(10.0, 80.0)` so default rigs never illuminate the
   object from below the ground plane; users may widen the range explicitly.
8. **Match repo conventions.** `from __future__ import annotations`;
   Google-style docstrings (`Args:` / `Returns:` / `Raises:`); frozen
   dataclasses and `Literal` types in the pure layer (no pydantic in
   `light_poses.py` — pydantic stays in `config.py`, mirroring how
   `camera_poses.py` is stdlib-only); keyword-only args. pip + hatchling
   only. `ruff check`, `ruff format --check`, and `mypy --no-sqlite-cache src`
   (strict) must stay clean.
9. **Coordinate with the backgrounds plan.** If
   `.ai/specs/randomized-backgrounds-04-06.md` has been (or is being)
   implemented, this plan's `render.py` changes land **alongside** that
   wiring, not instead of it — the two features are independent and must
   compose in the same render loop (per-frame: clear/build lights → move
   camera → render → composite background).

---

## 3. Atomic tasks

### T1 — Pure sampler: `light_poses.py` (+ small `camera_poses.py` refactor)

**Goal:** Deterministic per-frame light-rig sampling, bpy-free, reusing the
camera sampler's spherical math.
**Depends on:** none.
**Files:** create `src/rembrandt/light_poses.py`, `tests/test_light_poses.py`;
edit `src/rembrandt/camera_poses.py`, `tests/test_camera_poses.py` (only if an
import path changes).

**Steps:**

- In `camera_poses.py`, promote the private spherical helper to a public,
  documented function (keep behavior identical):

  ```python
  def position_from_spherical(
      *,
      azimuth: float,
      elevation: float,
      distance: float,
      look_at: Point3D,
  ) -> Point3D:
      """World-space point at spherical coordinates around ``look_at`` (radians)."""
  ```

  `_pose_from_spherical` becomes a thin wrapper over it (or call sites are
  updated directly). No sampling behavior may change — the existing
  reproducibility tests are the guard.
- Create `light_poses.py` (module docstring states it is pure Python with no
  Blender imports, same spirit as `camera_poses.py`), containing:

  ```python
  LightType: TypeAlias = Literal["POINT", "SUN", "AREA"]

  # Single source of truth for per-type default energy (T3 makes scene.py
  # import this; units: POINT/AREA in Watts, SUN unitless).
  DEFAULT_LIGHT_ENERGY: dict[LightType, float] = {
      "POINT": 1000.0,
      "SUN": 5.0,
      "AREA": 100.0,
  }

  @dataclass(frozen=True)
  class SampledLight:
      """One light in a per-frame rig, mirroring ``Scene.add_light`` params."""

      light_type: LightType
      location: Point3D
      look_at: Point3D
      energy: float
      color: Point3D
      size: float
  ```

- Implement the sampler:

  ```python
  def sample_light_rig(
      *,
      frame_index: int,
      count_range: tuple[int, int] = (1, 3),
      light_types: Sequence[LightType] = ("POINT", "SUN", "AREA"),
      azimuth_range: tuple[float, float] = (0.0, 360.0),
      elevation_range: tuple[float, float] = (10.0, 80.0),
      distance_range: tuple[float, float] = (4.0, 8.0),
      energy_scale_range: tuple[float, float] = (0.5, 2.0),
      color_jitter: float = 0.0,
      area_size_range: tuple[float, float] = (1.0, 3.0),
      look_at: Point3D = (0.0, 0.0, 0.0),
      seed: int | None = None,
  ) -> list[SampledLight]:
  ```

  Behavior, in this exact sampling order so determinism is well-defined:
  1. `rng = Random(seed + frame_index)` when `seed is not None`, else a fresh
     unseeded local `Random()`. `frame_index < 0` raises `ValueError`.
  2. `count = rng.randint(*count_range)`.
  3. For each light, in order: `light_type = rng.choice(light_types)`;
     elevation via **sin-elevation uniform** sampling (identical math to
     `_sample_random` — area-uniform on the band); azimuth and distance
     uniform in their ranges; position via `position_from_spherical`;
     `energy = DEFAULT_LIGHT_ENERGY[light_type] * rng.uniform(*energy_scale_range)`;
     color: each RGB channel `rng.uniform(1.0 - color_jitter, 1.0)`, then if
     `color_jitter > 0` divide all three by the max channel so jitter tints
     without systematically dimming; `size = rng.uniform(*area_size_range)`
     (sampled for **every** light regardless of type, so the rng stream — and
     therefore reproducibility — does not depend on which type was drawn;
     `Scene.add_light` already ignores `size` for non-AREA types).
  4. For SUN, only the direction `location → look_at` matters (document this
     in the docstring; the band position is still how the direction is
     sampled).
- Validation (raise `ValueError` with field-named messages, matching the
  `camera_poses` style): `count_range` ints with `1 <= min <= max`;
  `light_types` non-empty with only valid members; angular/distance ranges
  validated via the shared validator extracted in T2 — until T2 lands,
  validate locally with identical messages and switch over in T2;
  `energy_scale_range` and `area_size_range` positive with `min <= max`;
  `0.0 <= color_jitter <= 1.0`.

**Acceptance:**

- `pytest tests/test_light_poses.py -v` passes, covering at minimum (mirror
  the structure of `test_camera_poses.py`):
  - rig sizes always within `count_range` over many frames; with
    `count_range=(2, 2)` every rig has exactly 2 lights;
  - every sampled light's recovered azimuth/elevation/distance (reuse the
    `_recover_angles` test-helper pattern) falls within the requested ranges,
    including a non-origin `look_at`;
  - `light_type` always drawn from `light_types`; restricting to
    `("SUN",)` yields only SUN lights;
  - energy equals the type default × a factor inside `energy_scale_range`;
  - `color_jitter=0.0` → exact white `(1.0, 1.0, 1.0)`; `color_jitter=0.3` →
    all channels in `[0.7, 1.0]` and `max(channel) == 1.0` after
    normalization;
  - reproducibility: same `(seed, frame_index)` → equal rigs; different
    `frame_index` with the same seed → differing rigs across a multi-frame
    sample; `seed=None` works; global-RNG-non-mutation test;
  - rng-stream stability: with a fixed seed, rigs are identical whether or
    not any AREA light happens to be drawn earlier (guarded implicitly by the
    always-sample-size rule — add a regression test with
    `light_types=("POINT",)` vs mixed types only asserting self-consistency
    per configuration);
  - validation table-driven cases for every rule above;
  - `test_light_poses_module_is_bpy_free` source scan.
- All existing `test_camera_poses.py` tests pass unchanged (the helper
  promotion is behavior-neutral).
- `ruff` + `mypy` clean.

---

### T2 — Config schema: `LightRandomizationConfig` (+ shared range validator)

**Goal:** The YAML contract, default-off, with spherical-range validation
shared with the camera config instead of duplicated.
**Depends on:** T1.
**Files:** edit `src/rembrandt/config.py`, `src/rembrandt/camera_poses.py`,
`tests/test_config.py`.

**Steps:**

- In `camera_poses.py`, extract the azimuth/elevation/distance checks from
  `validate_camera_pose_inputs` into a public
  `validate_spherical_ranges(*, azimuth_range, elevation_range, distance_range)`
  and have `validate_camera_pose_inputs` call it (messages unchanged —
  existing `match=` assertions in tests are the guard).
- Add to `config.py`:

  ```python
  class LightRandomizationConfig(BaseModel):
      """Per-frame randomized light rigs. Off ("static") by default."""

      mode: Literal["static", "random"] = "static"
      count_range: tuple[int, int] = (1, 3)
      light_types: list[LightType] = Field(
          default_factory=lambda: ["POINT", "SUN", "AREA"]
      )
      azimuth_range: tuple[float, float] = (0.0, 360.0)
      elevation_range: tuple[float, float] = (10.0, 80.0)
      distance_range: tuple[float, float] = (4.0, 8.0)
      energy_scale_range: tuple[float, float] = (0.5, 2.0)
      color_jitter: float = Field(default=0.0, ge=0.0, le=1.0)
      area_size_range: tuple[float, float] = (1.0, 3.0)
      look_at: tuple[float, float, float] = (0.0, 0.0, 0.0)
      seed: int | None = None

      @model_validator(mode="after")
      def _check_ranges(self) -> Self:
          # delegate to validate_spherical_ranges + the T1 count/energy/size
          # checks so config and sampler can never disagree
          ...
  ```

  Implement `_check_ranges` by calling the same validation the sampler uses
  (the cleanest form: T1 exposes its parameter validation as a public
  `validate_light_rig_inputs(...)` and both the sampler and this model call
  it — mirroring the existing `CameraConfig` ↔ `validate_camera_pose_inputs`
  pattern).
- Add `light_randomization: LightRandomizationConfig =
  Field(default_factory=LightRandomizationConfig)` to `RembrandtConfig`.
- Docstrings: note that in `random` mode the static `lights:` list is
  **ignored** (not merged), that `seed` is independent of `camera.seed` and
  `background.seed`, and that `energy_scale_range` multiplies the per-type
  defaults in `light_poses.DEFAULT_LIGHT_ENERGY`.

**Acceptance:**

- `pytest tests/test_config.py -v` passes with new cases: defaults applied
  for configs without the block (`mode == "static"`); full round-trip
  dump→load equality with a populated block; table-driven rejection of
  invalid `count_range`, empty/invalid `light_types`, inverted or
  out-of-bounds angular ranges (same `match=` field names as the camera
  cases), non-positive `energy_scale_range`/`area_size_range`,
  `color_jitter > 1`, and `mode: disco`.
- Existing camera-validation tests pass unchanged after the extraction.
- bpy-free; `ruff` + `mypy` clean.

---

### T3 — `Scene.clear_lights()` + shared energy defaults

**Goal:** The single bpy change: per-frame rig swapping without leaking data
blocks, and `scene.py` consuming the shared energy-default constant.
**Depends on:** T1.
**Files:** edit `src/rembrandt/scene.py`; create or extend a bpy-marked test
(e.g. `tests/test_scene_lights.py`).

**Steps:**

- Add to `Scene`:

  ```python
  def clear_lights(self) -> None:
      """Remove all tracked lights (objects and data blocks) from the scene."""
      for light_obj in self.lights:
          light_data = light_obj.data
          bpy.data.objects.remove(light_obj, do_unlink=True)
          bpy.data.lights.remove(light_data)
      self.lights = []
      bpy.context.view_layer.update()
  ```

- In `add_light`, replace the inline
  `{"POINT": 1000.0, "SUN": 5.0, "AREA": 100.0}` dict with
  `DEFAULT_LIGHT_ENERGY` imported from `rembrandt.light_poses` (pure module —
  the import direction bpy-module → pure-module is allowed; the reverse is
  not). Update the `add_light` docstring's defaults sentence to reference it.

**Acceptance:**

- New bpy-marked tests (skip cleanly without bpy; fail under
  `--require-bpy`): after adding 3 lights of mixed types and calling
  `clear_lights()`, `scene.lights == []`, no `LIGHT`-type objects remain in
  `bpy.data.objects`, **and** `len(bpy.data.lights) == 0`; calling
  `clear_lights()` twice is a no-op; `Scene.clear()` behavior is unchanged;
  adding a light after clearing works and renders (reuse the smoke-test
  pattern at 64×64 if cheap).
- `add_light` with `energy=None` still applies 1000/5/100 per type (assert
  via the created light's `data.energy`).
- `pytest -m bpy --require-bpy -q` green; `ruff` + `mypy` clean.

---

### T4 — Wire per-frame rigs into `render_from_config`

**Goal:** Config-driven randomized lighting in the render loop; static mode
byte-for-byte unchanged.
**Depends on:** T1–T3.
**Files:** edit `src/rembrandt/render.py`, `tests/test_render_cli.py`.

**Steps:**

- In `render_from_config`:

  ```python
  randomize_lights = cfg.light_randomization.mode == "random"

  if not randomize_lights:
      for light in cfg.lights:          # existing behavior, unchanged
          scene.add_light(...)

  ...
  for index, pose in enumerate(poses):
      if randomize_lights:
          scene.clear_lights()
          rig = sample_light_rig(
              frame_index=index,
              **cfg.light_randomization.model_dump(exclude={"mode"}),
          )
          for light in rig:
              scene.add_light(
                  light_type=light.light_type,
                  location=light.location,
                  look_at=light.look_at,
                  energy=light.energy,
                  color=light.color,
                  size=light.size,
              )
      scene.move_camera(location=pose.location, look_at=pose.look_at)
      ...
  ```

  (If the field names of `LightRandomizationConfig` and `sample_light_rig`
  are kept identical — they must be — the `model_dump(exclude={"mode"})`
  expansion works directly; otherwise map explicitly.)
- Order within the frame matters and must be: lights → camera → render
  → (background composite, if that plan is merged). Extend the per-frame
  stdout line with the rig summary in `random` mode, e.g.
  `lights=[SUN, POINT]`.
- `render.py` still imports bpy only through `Scene`.

**Acceptance:**

- `pytest tests/test_render_cli.py -v` passes with new bpy-free cases using
  the existing `MagicMock` scene-factory pattern:
  - `mode: static` (and configs without the block): `clear_lights` **never**
    called; `add_light` called exactly `len(cfg.lights)` times, all **before**
    the first `move_camera`; all existing wiring assertions pass unchanged;
  - `mode: random` with `count_range: (2, 2)` and `n: 3` camera poses:
    `clear_lights` called exactly 3 times; `add_light` called exactly 6
    times; per frame, the call order is `clear_lights` → `add_light`×2 →
    `move_camera` → `render` (assert via `MagicMock.mock_calls` ordering);
  - determinism: two runs with the same `light_randomization.seed` produce
    identical `add_light` argument sequences; different seeds differ;
  - the static `lights:` list is ignored in `random` mode (give the config a
    distinctive static light and assert its params never reach `add_light`).
- `test_render_module_only_imports_bpy_through_scene` still passes.
- Optionally extend the bpy smoke test: 2 frames in `random` mode with a
  fixed seed render successfully and the two frames' pixel data differ
  (different rigs → different shading).
- `ruff` + `mypy` clean.

---

### T5 — Docs, sample config, frontend type mirror

**Goal:** Discoverability, and keep the SPA's default-config mirror honest.
**Depends on:** T1–T4.
**Files:** edit `README.md`, `configs/dataset.yaml` (or a commented example),
`frontend/src/types.ts`, `frontend/src/defaultConfig.ts`.

**Steps:**

- README:
  - extend the **Config Format** example with the `light_randomization`
    block (commented as optional, default `static`);
  - add a short **Randomized lighting** subsection: per-frame rigs, the
    sampled axes (count, type, spherical position, energy scale, color
    jitter, AREA size), determinism via `light_randomization.seed`, the
    "static `lights:` is ignored in `random` mode" rule, the SUN
    direction-only note, and that `energy_scale_range` multiplies per-type
    defaults (1000 W POINT / 5 SUN / 100 W AREA) because absolute energies
    are not comparable across types;
  - note the default elevation band `(10, 80)` deliberately keeps lights
    above the ground plane.
- Frontend (no UI controls, types only): add to `types.ts`

  ```ts
  export type LightRandomizationMode = "static" | "random";

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
  ```

  add `light_randomization?: LightRandomizationConfig` to `RembrandtConfig`,
  and include `light_randomization: { mode: "static" }` in
  `createDefaultConfig` so the "Defaults mirroring
  `rembrandt.config.RembrandtConfig`" comment stays true.

**Acceptance:**

- `cd frontend && yarn typecheck && yarn build` clean.
- SPA-saved configs still round-trip through `POST /api/config/save` and
  `load_config` (pydantic fills the new defaults; existing
  `test_save_config_writes_yaml` passes).
- The README example YAML validates via `load_config`.

---

## 4. Definition of done

- A config with

  ```yaml
  light_randomization:
    mode: random
    count_range: [1, 3]
    light_types: [POINT, SUN, AREA]
    elevation_range: [10.0, 80.0]
    energy_scale_range: [0.5, 2.0]
    color_jitter: 0.15
    seed: 7
  ```

  renders frames with visibly varied lighting (direction, intensity, warmth,
  softness) across the run; rerunning the same config reproduces the
  identical rig sequence.
- A config without the block (every existing config) behaves exactly as
  today: static lights added once, `clear_lights` never invoked, identical
  `add_light` call sequence.
- All sampling logic lives in bpy-free `light_poses.py`; the spherical math
  and range validation are shared with `camera_poses.py`, not duplicated; the
  per-type energy defaults exist in exactly one place and `scene.py` imports
  them.
- `scene.py` gained only `clear_lights()` (plus the constant import), and it
  provably leaks no light objects or data blocks across frames.
- `render.py` still imports bpy only through `Scene`, and composes with the
  randomized-backgrounds wiring if present (lights → camera → render →
  composite).
- `pytest -m "not bpy" -q`, `pytest -m bpy --require-bpy -q`, `ruff check src
  tests`, `ruff format --check src tests`, and `mypy --no-sqlite-cache src`
  are all clean.

## 5. Quick verification commands

```bash
pytest -m "not bpy" -q                       # fast lane incl. light_poses + config + wiring tests
pytest -m bpy --require-bpy -q               # clear_lights + render smoke
ruff check src tests && ruff format --check src tests
mypy --no-sqlite-cache src
cd frontend && yarn typecheck && yarn build

# end-to-end eyeball
rembrandt-render ./configs/dataset-random-lights.yaml
# flip through output/<stamp>/frame_*.png — lighting direction/intensity/tint
# should vary frame to frame; same seed twice → identical frames
```