
## 1. Extract the forward-vector-to-rotation primitive

**What you have.** In `scene.py`, the line `direction.to_track_quat("-Z", "Y").to_euler()` appears twice — once in `move_camera`, once in `add_light` for SUN/AREA lights. Both sites also do their own zero-length-direction guard (only the camera path does, actually — the light path will silently fail).

**What BlenderProc has.** A single helper in `CameraUtility.py`:

```python
def rotation_from_forward_vec(forward_vec, up_axis='Y', inplane_rot=None) -> np.ndarray:
    rotation_matrix = Vector(forward_vec).to_track_quat('-Z', up_axis).to_matrix()
    ...
    return np.array(rotation_matrix)
```

**Why it's better.** One source of truth for the `-Z` forward / `Y` up convention. If you ever change conventions (e.g., switch up axis for a different dataset format), it's one edit, not N. The zero-length check also lives in one place.

**Change.** Pull this into `rembrandt/camera/orientation.py` (or wherever your shared camera math will live), no bpy import needed. Replace both call sites in `scene.py`. The light path should reuse the same zero-length guard.

---

## 2. Use a derived FOV instead of `camera.data.angle_x` directly

**What you have.** `_fit_camera_to_target` reads FOV via `min(camera_obj.data.angle_x, camera_obj.data.angle_y)`.

**What BlenderProc has.** A dedicated `get_fov()` that computes FOV from the K-matrix-derived focal length and resolution, with this comment attached:

> Blender also offers the current FOV as direct attributes of the camera object, however at least the vertical FOV heavily differs from how it would usually be defined.

```python
def get_fov() -> Tuple[float, float]:
    K = get_intrinsics_as_K_matrix()
    fov_x = 2 * np.arctan(bpy.context.scene.render.resolution_x / 2 / K[0, 0])
    fov_y = 2 * np.arctan(bpy.context.scene.render.resolution_y / 2 / K[1, 1])
    return fov_x, fov_y
```

**Why it's better.** This is a correctness concern, not a style one. Blender's `angle_y` is derived from a sensor-fit assumption that doesn't always match what a user of `set_intrinsics_from_K_matrix` (or anyone setting a non-default sensor fit / pixel aspect / shift) would expect. The K-matrix-derived version always reflects the actual rendered image's FOV.

For your *current* code path — fixed sensor, no shift, square-ish renders — the two values likely agree. But the moment any future caller adjusts `sensor_fit`, `pixel_aspect_x/y`, or sets intrinsics via a K matrix, your fit math will silently use the wrong FOV.

**Change.** Add a small `get_fov(camera)` helper that computes from `K_matrix` + resolution, and have `_fit_camera_to_target` use it. (This also implies adding a `get_intrinsics_as_K_matrix(camera)` helper, which is a one-screen port of BlenderProc's version and is the prerequisite — also pure math, no bpy state mutation.)

---

## 3. Separate fit math from bpy mutation

**What you have.** `_fit_camera_to_target` reads bpy state (`bound_box`, `matrix_world`, `camera.data.angle_x`), computes the fit distance inline, and mutates `camera_obj.location` — all in one method.

**What BlenderProc has.** A clean split. Geometric/projective math sits in pure functions that take values and return values (`get_sensor_size(cam)`, `get_view_fac_in_px(...)`, `get_projection_matrix(...)`, `get_fov()`, `get_camera_frustum(...)`). Each one accepts inputs and returns numbers/arrays. The bpy mutation lives in setters like `set_intrinsics_from_blender_params`, which call the math functions and then apply results.

**Why it's better.** Two reasons:

- *Testability.* The fit calculation — given a radius, an FOV, and a margin, what distance fits? — is pure arithmetic. You can unit-test it without spinning up Blender. As written, you can't.
- *Reuse.* The same fit math is going to be useful in places that aren't "mutate this camera now" — for instance, validating a sampled distance, or reporting "the smallest distance that would frame this object" in dataset metadata. BlenderProc's split lets the math be called anywhere; yours doesn't.

**Change.** Refactor `_fit_camera_to_target` into:

```python
# pure, no bpy — testable
def fit_distance(*, target_radius: float, fov_rad: float, margin: float = 1.2) -> float:
    return (target_radius * margin) / sin(fov_rad / 2)
```

`_fit_camera_to_target` then reduces to (a) reading bpy state into plain values, (b) calling `fit_distance`, (c) applying the result. Same behavior, cleaner seams.

This is the architectural pattern that makes items 1 and 2 above easier — and it's the same seam BlenderProc maintains throughout `CameraUtility`.

---

## Order of work

All three are pure refactors with no behavior change:

1. **Item 1** — smallest, isolates the convention. Do first.
2. **Item 3** — extract the fit math. Sets up the pure-function layer.
3. **Item 2** — port `get_intrinsics_as_K_matrix` + `get_fov` into that layer and swap the FOV read. Catches a latent correctness issue.