Implement asset scale normalization in Rembrandt: imported OBJ geometry is scaled to a
canonical unit size so camera/light distance ranges and light energy defaults are
meaningful for any asset, regardless of its native scale. Read `.ai/AGENTS.md` first —
hard rules 1–3 apply throughout (bpy-free web/preview stack; Python is the source of
truth; one orientation/centering convention defined once in `convention.py`, with parity
between the pure preview path and the bpy render path).

## Problem being solved

All distances in the config are absolute world units (`camera.distance_range: [3, 5]`,
`light_randomization.distance_range: [4, 8]`, static light locations). Assets come in
arbitrary scales, so for a large OBJ the sampled lights sit inside or right at the object,
and POINT/AREA energy defaults (tuned for ~unit-sized objects) are wrong by inverse-square.
Fix: normalize the object once at import, in BOTH the preview and render paths, so that
its union bounding box has a half-diagonal of exactly 1.0 world unit.

## Canonical definition (single source of truth)

Add to `src/rembrandt/convention.py` (pure, NO bpy):

```python
def bounding_radius_from_bbox(bbox: npt.NDArray[np.float64]) -> float:
    """Half-diagonal of an axis-aligned bbox ``[[min], [max]]`` (shape (2, 3))."""
    half_extent = (bbox[1] - bbox[0]) / 2.0
    return float(np.linalg.norm(half_extent))
```

The normalization scale factor is defined as `s = target_radius / bounding_radius_from_bbox(bbox)`
with `target_radius = 1.0`, computed on the **union** bbox AFTER orient + center. This
definition is deliberately bbox-based (not max-vertex-distance) because both paths can
compute it identically and cheaply: the pure path from `orient_and_center`'s returned bbox,
the bpy path from the union of `_target_world_corners()`. Raise `ValueError("cannot
normalize degenerate geometry with zero bounding radius")` when the radius is 0.

Also add the pure transform used by the preview path:

```python
def normalize_scale(
    vertices: npt.NDArray[np.float64],
    bbox: npt.NDArray[np.float64],
    *,
    target_radius: float = 1.0,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float]:
    """Scale centered vertices and bbox about the origin to the target radius.

    Returns (scaled_vertices, scaled_bbox, scale_factor).
    """
```

(Pure multiplication: vertices * s, bbox * s. Must be called only on already-centered
geometry — document that in the docstring.)

## Config

In `src/rembrandt/config.py`, add to `ObjectConfig`:

```python
normalize: bool = True
```

Default True is intentional: this changes the meaning of existing configs that hand-author
absolute positions (static `lights[].location`, non-origin `camera.look_at`) — those are
now in normalized units where the object half-diagonal is 1. Note this in the field
docstring.

## Preview path (pure)

`src/rembrandt/preview/mesh.py`: `load_preview_mesh(path, *, up_axis, normalize: bool = True)`.
After the existing `orient_and_center` call, when `normalize` is true apply
`convention.normalize_scale` to vertices and bbox. No other geometry change.

`src/rembrandt/web/api.py`: `PreviewMeshRequest` gains `normalize: bool = True`; pass it
through to `load_preview_mesh`. No change to `/preview/poses` — it already receives the
(now normalized) bbox from the mesh response, so the band geometry scales automatically.

## Render path (bpy)

`src/rembrandt/scene.py`: add

```python
def normalize_target(self, *, target_radius: float = 1.0) -> float:
    """Uniformly scale all targets about the world origin to the canonical radius.

    Must be called after center_target(). Returns the applied scale factor.
    Raises RuntimeError when no target is loaded, ValueError for degenerate geometry.
    """
```

Implementation: compute the union bbox from `_target_world_corners()` (min/max over all
corners, same as `center_target` does), get the radius via
`convention.bounding_radius_from_bbox`, compute `s = target_radius / radius`, then for
every object in `self.targets` apply a uniform world-origin scale:

```python
scale_matrix = Matrix.Scale(s, 4)
for target in self.targets:
    target.matrix_world = scale_matrix @ target.matrix_world
bpy.context.view_layer.update()
```

Use the matrix form (NOT `target.scale *= s` / `target.location *= s` separately) so it is
correct for any per-object transform the importer produced. Import `Matrix` from
`mathutils` alongside the existing `Vector` import.

`src/rembrandt/render.py`: in `render_from_config`, after `scene.center_target()`:

```python
normalization_scale: float | None = None
if cfg.object.normalize:
    normalization_scale = scene.normalize_target()
```

Record it in the run metadata: add `"normalization_scale": normalization_scale` to the
payload built by `_run_metadata_payload` (top level, next to `resolved_object_path`).
Thread it through the worker-partial path too so coordinator-merged `run.json` carries it
(simplest: pass it into `merge_run_metadata` from the coordinator by recomputing is NOT
possible without bpy — instead have each worker include it in its partial payload and have
`merge_run_metadata` take it from the first partial; assert all partials agree).

## Frontend (display only — no logic)

- `frontend/src/types.ts`: `ObjectConfig` gains `normalize?: boolean`.
- `frontend/src/defaultConfig.ts`: `normalize: true`.
- `frontend/src/api.ts`: `fetchMesh` request body gains `normalize`.
- `frontend/src/controls/Controls.tsx`, Object section: a checkbox row
  "Normalize to unit size" bound to `config.object.normalize`, defaulting checked, with a
  hint line: "Distances and light energy assume a unit-sized object. Disable only if your
  config uses the asset's native units." Changing it should trigger a mesh reload the same
  way changing the up-axis does (mirror the existing `onObjectUpAxisChange` wiring with an
  `onObjectNormalizeChange` handler in `App.tsx` that refetches the mesh and poses).

## Tests

Pure lane (`pytest -m "not bpy"`):
- `tests/test_convention.py`:
  - `test_bounding_radius_from_bbox_half_diagonal` — bbox [[-1,-2,-3],[1,2,3]] → radius
    sqrt(14).
  - `test_normalize_scale_unit_radius` — after normalize, recomputed half-diagonal == 1.0
    (within 1e-12) and returned scale is correct.
  - `test_normalize_scale_rejects_degenerate` — single repeated vertex → ValueError.
- `tests/test_preview_mesh.py`: loading a generated OBJ with known extent at
  `normalize=True` yields bbox half-diagonal 1.0; `normalize=False` preserves raw extent.
- `tests/test_web_api.py`: `/preview/mesh` honors the `normalize` flag both ways.
- `tests/test_render_cli.py`: `run.json` contains `normalization_scale` (mocked scene:
  make the MagicMock's `normalize_target.return_value = 0.5` and assert it lands in the
  payload; also assert `normalize_target` is NOT called when `object.normalize` is false).

bpy lane (`pytest -m bpy --require-bpy`):
- Extend `tests/test_orientation_parity.py`: for the existing fixtures AND the two-offset-
  cubes case, run preview (`normalize=True`) vs `Scene.load_object + center_target +
  normalize_target` and assert vertex sets allclose (reuse `_assert_vertex_sets_allclose`)
  and union bbox half-diagonal == 1.0 in both paths (atol 1e-5). This is the new parity
  guard: orientation + centering + scale may never drift between paths.
- `tests/test_scene_camera_fit.py`-adjacent: after normalize, `target_radius_about((0,0,0))`
  == 1.0 within 1e-5.

## Documentation

- `.ai/AGENTS.md`: hard rule 3 becomes "One orientation/centering/**scale** convention,
  defined once" — normalization is part of the canonical frame; both paths derive from
  `convention.py`; parity tests cover scale.
- `README.md` config section: document `object.normalize` (default true), what "unit size"
  means (union-bbox half-diagonal = 1), that distance ranges and light positions are in
  normalized units, and the opt-out for configs authored in native asset units.
- `configs/dataset.yaml`: add `normalize: true` under `object:`.

## Constraints

- `convention.py`, `preview/`, `web/`, `config.py` stay bpy-free (the source-scanning
  tests enforce this — do not break them).
- Do not change `camera_poses.py`, `light_poses.py`, `framing.py`, or any default
  distance/energy values — the entire point is that existing defaults become correct.
- Do not rescale labels/masks/postfx — they are 2D and unaffected.

## Acceptance

All of: `pytest -m "not bpy" -q`, `pytest -m bpy --require-bpy -q`, `ruff check`,
`ruff format --check`, `mypy --no-sqlite-cache src`, `cd frontend && yarn typecheck &&
yarn build`. Manual: load a large OBJ in the SPA — the camera band now wraps the object
proportionally; render a config with `light_randomization.mode: random` — lights sit in a
sensible shell around the object instead of inside it.