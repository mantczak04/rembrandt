# Rembrandt — Orientation Bug: Fresh-Agent Handoff & Fix Plan

> **You are picking this up cold.** This document is self-contained: it explains the
> project, the bug, the *confirmed* root cause, what has already been tried and what is
> already committed, the hard constraints, and a concrete plan to land the fix. A prior
> agent investigated thoroughly but did not close it. Your job is to **fix** it, not
> re-investigate from scratch — though you must verify the diagnosis holds before
> changing code.
>
> **One discipline above all:** Blender's `obj_import` axis behavior and the matching
> pure-Python rotation are easy to get wrong by a sign or an axis swap. **Do not reason
> the matrix out on paper and trust it.** Use the bpy parity test and an actual rendered
> frame as your oracles for correctness. Change one thing, prove it, repeat.

---

## 1. What Rembrandt is (minimum you need)

Rembrandt generates synthetic computer-vision datasets from a 3D model. The user supplies
a Wavefront `.obj`, configures camera coverage + lighting, and Rembrandt renders many
images of the object from sampled camera poses on a sphere around it.

It has two cooperating tools that share **one YAML config** as their contract:

- **`rembrandt-serve`** — a bpy-free FastAPI server + React/Three.js SPA. The user pastes
  an `.obj` path and previews, in an orbit-controllable 3D view, the sampled camera band
  wrapped around the oriented object. It is a *configuration-sanity* tool. It never
  renders images or runs Blender.
- **`rembrandt-render CONFIG_PATH`** — the Blender/`bpy` pipeline that reads the YAML and
  writes PNG frames.

**Core design invariant:** both tools must place the object in the **same canonical
frame**. That transform (axis orientation + center-on-bbox) is defined once in
`src/rembrandt/convention.py` and is used by both paths — the preview via the pure
function `orient_and_center()`, and the render via the axis constants `OBJ_IMPORT_*`
passed to `bpy.ops.wm.obj_import` in `src/rembrandt/scene.py`.

Rembrandt's camera sampler treats **+Z as world up** (elevation is measured from the XY
plane; `z = distance * sin(elevation)`). So an object must end up **standing upright on
+Z** in the canonical frame for the camera band to make sense.

---

## 2. The bug (symptom)

The rendered object orientation is wrong. The user's test asset is a tall chess **pawn**.
In rendered frames the pawn does **not** stand upright (head/ball up, base down on +Z) —
it lies on its **side**, long axis horizontal. (The user's first glance called it
"upside down"; the precise, reproduced behavior is "on its side." Both are the same
underlying defect.)

The frontend preview *looked* upright to the user, but that was an artifact of the orbit camera
angle they happened to view it from — **the preview geometry is mis-oriented in exactly
the same way as the render.** Do not assume the current preview is already correct. For
this asset, both paths are wrong and they agree with each other.

---

## 3. Confirmed root cause

**Rembrandt hard-codes a "source OBJ is Y-up" assumption, but the asset is authored
Z-up.** Applying a Y-up→Z-up rotation to an already-Z-up mesh lays it on its side.

Evidence gathered (on `test-obj/chess.obj` — its MTL references
`12931_WoodenChessPawnSideA` and the mesh is a wooden pawn):

| | Raw OBJ | After `orient_and_center` / `obj_import` |
|---|---|---|
| Long axis (head→base) | **Z**, extent 8.03 | **Y**, extent 8.03 |
| Short axes | X, Y ≈ 4.47 | X, Z ≈ 4.47 |

The mesh's long (head-to-base) axis is **Z** in the file — i.e. it is **Z-up native**.
Rembrandt's convention assumes Y-up and applies a +90° rotation about X (and `obj_import`
is called with `up_axis="Y"`), which sends the file's Z onto **world Y**. Result: the
pawn's long axis lands on world Y → it lies on its side in a +Z-up scene.

**Why preview and render agree (parity passes) yet the object is still wrong:** both paths
apply the *same* Y-up assumption to a Z-up asset. They are consistently wrong. Parity only
proves the two paths agree with each other — **not** that the result is upright.

**Why the existing tests give false green** (this is important — do not trust them as-is):

- `test_orient_and_center_matches_bpy_import*` only compares the two geometry paths to
  each other. Both wrong + agreeing → passes.
- `test_rendered_view_keeps_world_z_upright_on_chess_board_object` checks **world-Z**
  quantiles. For this asset the head/base sit on **world-Y** after the bad transform, so
  the test is measuring the wrong axis and passes even though the pawn is sideways.

**Reproduction on disk:** a prior run produced
`output/_orientation_repro/repro/frame_0000.png` using the dataset-style band (seed=42);
the pawn lies on its side. Reproduce this yourself before and after your fix.

### Branch verdicts (from investigation)

| Branch | Verdict |
|---|---|
| A — import drifts from `orient_and_center` | **Ruled out.** The two geometry paths match. |
| A′ — wrong source-up convention for a Z-up OBJ | **This is the cause.** Both paths apply the Y-up assumption to a Z-up asset. |
| B — camera orientation (`to_track_quat`) | Not the primary cause; geometry is already wrong with matched paths. Re-check only *after* A′ is fixed. |

---

## 4. Files to read first

- `src/rembrandt/convention.py` — the single source of truth: `_Y_UP_TO_Z_UP` matrix,
  `OBJ_IMPORT_FORWARD_AXIS` / `OBJ_IMPORT_UP_AXIS` constants, `orient_and_center()`.
- `src/rembrandt/scene.py` — `load_object()` (calls `bpy.ops.wm.obj_import` with those
  constants), `center_target()`, `move_camera()` / `_point_camera_at()`.
- `src/rembrandt/config.py` — pydantic config schema (`RembrandtConfig`, `ObjectConfig`,
  `CameraConfig`, …) and YAML load/dump. This is where a new `object` field would go.
- `src/rembrandt/preview/mesh.py` — preview path; calls `orient_and_center`.
- `src/rembrandt/camera/orientation.py` — `rotation_euler_from_forward` /
  `to_track_quat("-Z", "Y")` (relevant only if Branch B reappears).
- `src/rembrandt/render.py` — the render loop.
- `tests/test_convention.py`, `tests/test_orientation_parity.py` — the parity tests and
  the misleading "upright" test.
- `.ai/AGENTS.md` and `.ai/specs/spa-migration.-29-05.md` (Guardrail 3 + task T2) for the
  single-convention design intent.

---

## 5. Hard constraints (do not violate)

1. **One convention, in `convention.py`.** The preview transform and the `obj_import`
   axis settings must be derived from the **same** source-orientation declaration so they
   can never disagree. Do **not** add a compensating rotation in `render.py`,
   `scene.load_object`, or the frontend.
2. **bpy-free boundary is unchanged.** `convention.py`, `preview/*`, `web/*`, `config.py`
   must not import bpy (there are source-scanning guard tests enforcing this). bpy stays
   in `scene.py`, `camera/orientation.py` (lazy import), and the render entry only.
3. **No orientation math in the frontend.** The SPA only displays numbers the backend
   produces.
4. **Do not auto-detect up-axis from geometry shape.** "Longest extent = up" is wrong for
   many real objects (a board is flat; a car is long horizontally; this pawn happens to
   be tall). Up-axis must be an **explicit, declared** property, not inferred.
5. **Preserve existing Y-up behavior.** The fix must default to today's Y-up handling so
   genuinely Y-up assets and all current tests keep working unchanged.
6. **The preview is not assumed already-correct for this asset.** The goal is that the
   pawn stands upright on +Z in **both** paths and they still agree — not to make render
   match a preview that is itself sideways.
7. **Empirical oracle, not analysis.** Correctness is decided by (a) the bpy parity test
   passing for each supported source orientation and (b) an actual rendered frame showing
   the pawn upright. Never ship a matrix/constants change justified only by hand-derivation.
8. **Match repo conventions.** `from __future__ import annotations`; Google-style
   docstrings (`Args:`/`Returns:`/`Raises:`); `Literal` for enum-like params; keyword-only
   args where surrounding code uses them. `ruff check`, `ruff format --check`, and
   `mypy --no-sqlite-cache src` must stay clean (strict mode).

---

## 6. Already in the repo (do not redo)

A prior agent added the testing/CI scaffolding; keep it, build on it:

- `tests/conftest.py` — a `--require-bpy` flag that turns bpy-marked skips into failures
  (CI lane only; local runs without bpy still skip).
- `.github/workflows/ci.yml` — two lanes: `test-pure` (lint + `pytest -m "not bpy"`) and
  `test-bpy` (`pytest -m bpy --require-bpy`, with bpy wheel caching).
- `tests/test_orientation_parity.py` — preview API geometry vs scene geometry.
- `tests/test_paths.py` + `tests/fixtures/asymmetric_y_up.obj` — a CI-safe sample that
  prefers `test-obj/` assets when present.
- New (this round, **passing but misleading** — you will fix their assertions in T4):
  `test_orient_and_center_matches_bpy_import_on_chess_board_object`,
  `test_rendered_view_keeps_world_z_upright_on_chess_board_object`.
- `README.md` / `.ai/AGENTS.md` document the two-lane test story.

The asset `test-obj/chess.obj` must be present for the bpy
orientation tests. If it is not in the working tree, obtain it (the user has it locally)
and place it under `test-obj/` before running the bpy lane.

---

## 7. Fix plan (work in order; each task gates the next)

### T1 — Reproduce and pin the target state
**Goal:** See the failure with your own eyes and define "correct."
**Steps:**
- In the bpy environment, render a frame from the dataset-style band (seed=42) on the
  pawn asset (mirror `output/_orientation_repro`). Confirm the pawn lies on its side.
- Inspect the raw OBJ vs the post-transform geometry; confirm the table in §3 (long axis
  is Z raw, becomes Y after the current transform).
- Define the target: the pawn's head→base (long) axis lands on **world Z**, head above
  base, base near the ground plane; preview and render agree.
**Acceptance:** you can articulate, with measured axis extents, the current (wrong) and
target (upright) orientations.

### T2 — Make source up-axis a first-class, single-sourced convention
**Goal:** Stop hard-coding "Y-up source." Let the object declare its native up-axis, and
derive **both** the pure preview rotation and the `obj_import` axis pair from that one
declaration, in `convention.py`.
**Steps:**
- Add an explicit source-orientation field to the config — e.g. `object.up_axis:
  Literal["Y", "Z"] = "Y"` in `ObjectConfig` (`config.py`). Default `"Y"` to preserve
  current behavior. (Thread `forward` too if a pair is needed; keep the surface minimal.)
- In `convention.py`, replace the single hard-coded `_Y_UP_TO_Z_UP` matrix and the single
  `OBJ_IMPORT_*` pair with a mapping from the declared source up-axis to **both**:
  (a) the pure rotation matrix used by `orient_and_center`, and (b) the
  `(forward_axis, up_axis)` pair passed to `obj_import`. Define them **together** per
  orientation so they cannot drift. `orient_and_center` takes the up-axis as a parameter.
- Update `preview/mesh.py` and `scene.load_object` to pass the declared up-axis through.
  Do not branch on asset names; branch only on the declared orientation.
- For `up_axis="Z"`: the file's Z is already world up, so the head→base axis must remain
  on world Z. Determine the exact `(forward, up)` pair for `obj_import` that achieves this
  **and** the matching pure matrix — verified by parity (T3/T4), not by derivation.
**Acceptance:** `up_axis` flows from config → both paths via `convention.py` only; bpy-free
boundary intact; `ruff`/`mypy` clean.

### T3 — Apply Z-up to the pawn and verify end-to-end
**Goal:** The actual reported asset renders upright.
**Steps:**
- Set the pawn's config (or its test config) to `object.up_axis: "Z"`.
- Re-render the seed=42 band. Confirm the pawn stands upright (head→base on +Z, base near
  ground, not on its side).
- Confirm the preview path produces the same upright geometry (orbit-independent: check
  vertices/bbox, not a screenshot).
**Acceptance:** rendered frame shows an upright pawn; preview geometry matches the scene
geometry for the Z-up asset.

### T4 — Fix the misleading tests; add tests that actually catch this
**Goal:** Ensure the suite would have failed on the sideways pawn and now passes on the
upright one — for both source orientations.
**Steps:**
- Rewrite `test_rendered_view_keeps_world_z_upright_on_chess_board_object` (and any
  geometry "upright" check) so it asserts the **head→base / long axis lands on world Z**
  and head sits above base — i.e. it would **fail** on the pre-fix sideways geometry.
  Measuring world-Z quantiles is only valid once the long axis is actually on Z, which is
  the property under test.
- Add a parity test for the **Z-up** path (`orient_and_center(up_axis="Z")` vs
  `obj_import` with the Z-up axis pair), alongside the existing Y-up parity test. Keep
  both bpy-marked.
- Keep a Y-up regression (existing fixture / synthetic mesh) green to prove default
  behavior is unchanged.
**Acceptance:** `pytest -m "not bpy" -q` and `pytest -m bpy --require-bpy -q` both pass;
the new orientation assertions fail if you temporarily revert T2/T3.

### T5 — Re-check residual preview≠render (camera / Branch B)
**Goal:** Confirm nothing else remains once geometry is upright.
**Steps:**
- With the pawn now upright, render a few band poses and confirm the object reads as
  upright across them. Only if a residual flip/roll persists, investigate
  `rotation_euler_from_forward` / `to_track_quat("-Z", "Y")` (camera up should resolve
  toward world +Z for the band) and add a camera-up unit test.
**Acceptance:** band renders show a consistently upright pawn; no camera change needed, or
if one is, it is covered by a test.

---

## 8. Definition of done

- The pawn asset renders **upright on +Z** (head→base axis vertical, base near the ground
  plane), and the preview produces the same orientation.
- Source up-axis is an explicit config field, defaulting to `"Y"`; both the preview
  transform and `obj_import` derive from it via `convention.py` alone — no second
  orientation path, no per-call fixups, bpy-free boundary intact.
- Orientation tests now **fail** on the pre-fix sideways geometry and **pass** after the
  fix, for both Y-up and Z-up sources; parity holds for each.
- All existing Y-up behavior and tests are unchanged.
- `pytest -m "not bpy" -q`, `pytest -m bpy --require-bpy -q`, `ruff check`,
  `ruff format --check`, and `mypy --no-sqlite-cache src` are all clean.

---

## 9. Quick local verification commands

```bash
# pure lane (no Blender needed)
pytest -m "not bpy" -q
ruff check src tests && ruff format --check src tests && mypy --no-sqlite-cache src

# bpy lane (requires Blender/bpy + the pawn asset under test-obj/)
pytest -m bpy --require-bpy -q

# render a repro frame to eyeball orientation before/after the fix
rembrandt-render ./configs/<pawn-config>.yaml   # inspect output/<stamp>/frame_0000.png
```