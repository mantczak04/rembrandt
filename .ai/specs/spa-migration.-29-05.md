# Rembrandt — SPA Preview Pivot: Implementation Plan

> **For the implementing agent (Claude Code):** Read this whole file first, then
> `.ai/AGENTS.md`, `.ai/specs/camera-poses-13-05.md`,
> `.ai/specs/camera-and-scene-bugfix-and-optimization-28-05.md`, and the existing
> sources under `src/rembrandt/` and `scripts/camera_pose_preview.py`. Execute the
> tasks **in order** — each lists its dependencies, files, steps, and acceptance
> criteria. Do not start a task until its dependencies pass their acceptance gates.

---

## 1. Overview

### What we're building
Rembrandt currently renders synthetic object-detection datasets from a single
`.obj`, driven from `main.py` / a future CLI, with a Streamlit + Plotly camera-pose
preview as a side tool. We are pivoting the **primary interface** to a local
single-page app:

- A **browser SPA** (React + TypeScript + Vite + Three.js) that lets the user pick
  an object, tune the camera-sphere configuration with **live 3D feedback**, and
  **save a config file**.
- A **small FastAPI server** that backs the SPA. It is **bpy-free** — its only jobs
  are serving preview data (an oriented mesh + the camera angle band) and writing
  the config to disk.
- The **actual render stays a separate step**: `rembrandt render <config.yaml>`,
  which is the existing bpy pipeline. The SPA never triggers or monitors renders.

The two tools share **one YAML config** as their contract.

### Why
The preview's purpose is **not** to preview rendered images. It is a
**configuration-sanity tool**: the user looks at the camera angle band relative to
the object and decides whether the angular coverage is physically sensible (e.g. for
a chess piece sitting on a board, cameras below the board are useless — you'd clamp
elevation to exclude them). It answers "do these angles make sense for this object?"
before a slow render is ever run.

### Target shape
```
rembrandt serve            # FastAPI (bpy-free) + SPA on localhost — configure + preview
rembrandt render cfg.yaml  # bpy render loop (existing pipeline) — execute
```

```
rembrandt/
├── pyproject.toml          # + fastapi, uvicorn, pydantic, pyyaml, typer, rich; [project.scripts]
├── frontend/               # NEW — Vite + React + TS + Three.js SPA
│   ├── package.json
│   ├── vite.config.ts      # dev proxy /api -> uvicorn
│   ├── tsconfig.json
│   └── src/
├── src/rembrandt/
│   ├── camera_poses.py     # REUSE — pose sampling (do not reimplement in JS)
│   ├── scene.py            # REUSE — bpy render scene
│   ├── camera/             # REUSE — fit / intrinsics / orientation math
│   ├── errors.py           # REUSE
│   ├── config.py           # NEW — pydantic config schema + YAML load/dump
│   ├── convention.py       # NEW — single source of truth for axis + centering
│   ├── preview/            # NEW — bpy-free preview data (mesh parse, band geometry)
│   │   ├── mesh.py
│   │   └── geometry.py
│   ├── web/                # NEW — FastAPI app, API routes, serve entrypoint
│   │   ├── app.py
│   │   ├── api.py
│   │   └── serve.py
│   └── render.py           # NEW — config-driven `rembrandt render` (from main.py)
└── tests/
```

### In scope
The interface pivot: config schema, the bpy-free preview backend, the SPA, the
shared orientation convention, and turning `main.py` into a config-driven render
command.

### Out of scope (do **not** build here)
The render-pipeline internals that live *behind* `rembrandt render` and are already
on the `AGENTS.md` roadmap: 3D→2D bbox projection, 2D augmentations, YOLO label
writing, `data.yaml`, train/val split, the training-script template. Leave config
fields reserved for them, but do not implement them in this plan.

---

## 2. Guardrails (apply to every task)

1. **Python is the single source of truth. The frontend only displays.** The SPA
   computes nothing about poses, band geometry, orientation, or centering — it
   renders numbers the backend hands it. Never port `sample_camera_poses` or the
   orientation transform to JS.
2. **The web server is bpy-free.** Nothing under `src/rembrandt/web/`,
   `src/rembrandt/preview/`, `config.py`, or `convention.py` may `import bpy` or
   import any module that does. bpy lives **only** in `scene.py`, `camera/orientation.py`
   (lazy), and `render.py`.
3. **One object-orientation/centering convention, shared.** The transform that puts
   the object in its canonical frame (axis conversion + center on bbox-center) is
   defined **once** in `convention.py` and used by *both* the preview mesh endpoint
   and the bpy import in `scene.py`. They must not drift.
4. **Single object input source.** The user pastes a filesystem path. The backend
   reads that path for both the preview mesh and (later) the render. No file upload.
5. **The preview is about angles.** Distance is cosmetic (a display radius so the
   band wraps the object legibly). Do **not** apply camera-fit / framing math in the
   preview — that only affects distance, which the preview does not care about.
6. **Match existing conventions.** `from __future__ import annotations`; Google-style
   docstrings (`Args:`/`Returns:`/`Raises:`); `Literal` for enum-like params;
   keyword-only args where `scene.py`/`camera_poses.py` use them. All new pure modules
   get tests. Code must pass `ruff check`, `ruff format --check`, and `mypy` in strict
   mode (already configured in `pyproject.toml`).
7. **Test isolation.** Pure modules (`config`, `convention`, `preview/*`, `web/*` via
   TestClient) must be testable **without** bpy. Only `render.py` / `scene.py` tests
   may import bpy.

---

## 3. Atomic tasks

### T1 — Config schema (`config.py`)
**Goal:** Define the YAML contract both tools share.
**Depends on:** none.
**Files:** create `src/rembrandt/config.py`, `tests/test_config.py`; edit `pyproject.toml`.
**Steps:**
- Add deps `pydantic>=2`, `pyyaml` to `[project.dependencies]`.
- Define pydantic v2 models for the **complete** render config:
  - `object`: `path: str`.
  - `camera`: the `sample_camera_poses` params — `n`, `azimuth_range`,
    `elevation_range`, `distance_range`, `strategy` (`Literal["random","fibonacci"]`),
    `seed`, `look_at`. Mirror the validation already in `camera_poses._validate_inputs`
    via pydantic validators (reuse, don't duplicate the messages gratuitously).
  - `lights`: list of light specs (type/location/look_at/energy/color/size) matching
    `Scene.add_light` params. Provide sensible defaults.
  - `render`: `focal_length`, `resolution`, `engine` (`Literal["EEVEE","CYCLES"]`),
    `samples`.
  - `output`: `dir`, `train_val_split` (default `0.8`) — **reserved**, not consumed
    here.
- Add `load_config(path) -> RembrandtConfig` and `dump_config(cfg, path) -> None`
  (YAML via PyYAML). bpy-free.
**Acceptance:**
- `pytest tests/test_config.py -v` passes (round-trip dump→load equality; invalid
  ranges rejected with clear messages; defaults applied).
- `ruff check src/rembrandt/config.py tests/test_config.py` and `mypy src/rembrandt/config.py` clean.
- No `import bpy` anywhere in the module.

---

### T2 — Canonical object convention (`convention.py`)
**Goal:** One definition of "put the object in its canonical frame" so preview and
render agree. This is the single most drift-prone piece — be careful.
**Depends on:** none.
**Files:** create `src/rembrandt/convention.py`, `tests/test_convention.py`.
**Steps:**
- Define the canonical convention explicitly: the sampler treats **+Z as up**
  (`z = distance * sin(elevation)`), so the object's visual up must align to +Z, and
  the object is centered on its **bounding-box center** (matching `Scene.center_target`).
- Export the axis choice as named constants (the `forward_axis`/`up_axis` values to
  pass to `bpy.ops.wm.obj_import`) **and** a pure-Python function:
  `orient_and_center(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]`
  returning oriented+centered vertices and the bbox (min/max), bpy-free.
- The pure transform must reproduce what Blender's import does under the chosen axes
  (Y-up→Z-up is a fixed rotation about X). Keep the rotation and the bbox-centering in
  this one module.
- Refactor `Scene.load_object` (T8) to pass these same axis constants to `obj_import`
  — referenced here so the convention has a single home.
**Acceptance:**
- `pytest tests/test_convention.py -v` passes: a parity test that loads a small
  `.obj` (use `test-obj/`) through bpy import **and** through `orient_and_center`,
  asserting matching vertex sets within tolerance. Mark this test as bpy-dependent
  (it may import bpy); the `orient_and_center` unit tests themselves must not.
- `ruff` + `mypy` clean on the pure function.

---

### T3 — OBJ mesh parsing for preview (`preview/mesh.py`)
**Goal:** Turn a `.obj` path into oriented, centered, serializable geometry — bpy-free.
**Depends on:** T2.
**Files:** create `src/rembrandt/preview/__init__.py`, `src/rembrandt/preview/mesh.py`,
`tests/test_preview_mesh.py`.
**Steps:**
- Port the `.obj` parsing from `scripts/camera_pose_preview.py`
  (`_load_mesh`, `_parse_obj_index`, `_triangulate`) into `mesh.py`. Parse vertices +
  triangulated faces.
- Apply `convention.orient_and_center` (T2).
- Return a `PreviewMesh` dataclass: flat `positions: list[float]` (xyz) and
  `indices: list[int]` (triangles) ready for a Three.js `BufferGeometry`, plus `bbox`.
- Raise `ModelFileNotFoundError` (existing `errors.py`) for a missing path; raise a
  clear `ValueError` for a vertex-less file.
**Acceptance:**
- `pytest tests/test_preview_mesh.py -v` passes (parses `test-obj/` sample; vertex
  count > 0; bbox center ≈ origin after centering; missing-path raises).
- bpy-free; `ruff` + `mypy` clean.

---

### T4 — Preview band + points + ground plane (`preview/geometry.py`)
**Goal:** Compute the angular band, sampled camera points, and ground plane the SPA
draws — bpy-free, reusing the sampler.
**Depends on:** T3 (for bbox), reuses `camera_poses.sample_camera_poses`.
**Files:** create `src/rembrandt/preview/geometry.py`, `tests/test_preview_geometry.py`.
**Steps:**
- Port the band-surface and band-edge geometry from
  `scripts/camera_pose_preview.py` (`_sphere_surface`, `_add_band_edges`,
  `_spherical_to_cartesian`) into pure data builders (return vertex arrays / line
  point lists, **not** Plotly traces).
- Camera points: call `sample_camera_poses` with the config's camera params; return
  the `location`/`look_at` lists. (These are illustrative; do not apply fit math.)
- Ground plane: a quad at the object's bbox **base** (`bbox.min.z`), sized to the
  band radius — this is what makes "the band dips under the object" visible.
- Band display radius is cosmetic: derive it from the bbox extent / distance range
  for legibility; it does not change the angles.
**Acceptance:**
- `pytest tests/test_preview_geometry.py -v` passes (point count == `n`; band
  vertices within the requested azimuth/elevation extents; ground plane z == bbox
  base).
- bpy-free; `ruff` + `mypy` clean.

---

### T5 — FastAPI app skeleton (`web/app.py`)
**Goal:** The bpy-free web application object.
**Depends on:** none (endpoints wired in T6).
**Files:** create `src/rembrandt/web/__init__.py`, `src/rembrandt/web/app.py`,
`tests/test_web_app.py`; edit `pyproject.toml`.
**Steps:**
- Add deps `fastapi`, `uvicorn[standard]`.
- `create_app() -> FastAPI`: app factory. Mount an `/api` router (T6). In dev allow
  CORS from the Vite origin. Serve the built SPA: `StaticFiles` at `/` from the
  frontend build dir if present, with an `index.html` fallback for SPA client routing.
- Add `GET /api/health -> {"status": "ok"}`.
**Acceptance:**
- `pytest tests/test_web_app.py -v` passes using `fastapi.testclient.TestClient`
  (`/api/health` returns ok). No bpy import. `ruff` + `mypy` clean.

---

### T6 — Preview + config API endpoints (`web/api.py`)
**Goal:** The three endpoints the SPA calls.
**Depends on:** T1, T3, T4, T5.
**Files:** create `src/rembrandt/web/api.py`, `tests/test_web_api.py`.
**Steps:**
- `POST /api/preview/mesh` body `{path}` → `PreviewMesh` (T3) as JSON.
- `POST /api/preview/poses` body `{camera params}` → band + points + ground plane (T4).
- `POST /api/config/save` body `{config, filename}` → validate via `RembrandtConfig`
  (T1), `dump_config` into a `./configs/` dir **relative to the server's working
  directory** (create it if missing; reject path traversal in `filename`); return the
  written path.
- Use pydantic request/response models. Map domain errors
  (`ModelFileNotFoundError`, `ValueError`) to 4xx with a clear message.
**Acceptance:**
- `pytest tests/test_web_api.py -v` passes via TestClient: mesh endpoint returns
  positions/indices for the sample `.obj`; poses endpoint returns `n` points; save
  endpoint writes a valid YAML that `load_config` reads back; bad path → 4xx.
- bpy-free; `ruff` + `mypy` clean.

---

### T7 — `rembrandt serve` entrypoint (`web/serve.py`)
**Goal:** One command to launch the configurator.
**Depends on:** T5, T6.
**Files:** create `src/rembrandt/web/serve.py`; edit `pyproject.toml`.
**Steps:**
- `serve(host="127.0.0.1", port=8000, open_browser=True)`: run uvicorn on
  `create_app()`; optionally open the browser at the URL.
- Register `[project.scripts]` `rembrandt-serve = "rembrandt.web.serve:main"` (or a
  `rembrandt` group command if you add typer in T8 — keep consistent).
**Acceptance:**
- `rembrandt-serve` starts the server; `GET /api/health` responds; serves the SPA
  index when a build is present. `ruff` + `mypy` clean.

---

### T8 — Config-driven render command (`render.py`)
**Goal:** Turn `main.py` into `rembrandt render <config.yaml>`.
**Depends on:** T1, T2.
**Files:** create `src/rembrandt/render.py`, `tests/test_render_cli.py`; edit
`pyproject.toml`; update/retire `main.py`.
**Steps:**
- Add `typer` + `rich` (per `AGENTS.md` tooling). `render(config_path)`:
  `load_config` → build `Scene` → `load_object` (ensure it passes the **T2 axis
  constants** to `obj_import`) → `center_target` → add lights from config → add camera
  → loop `sample_camera_poses(**config.camera)` → `move_camera` → `render` each frame
  to `output/<stamp>/frame_XXXX.png`. Keep the per-frame stdout progress line.
- Register `[project.scripts]` `rembrandt-render = "rembrandt.render:main"`.
- **Out of scope:** bbox/YOLO/augment/split — leave the loop emitting frames only.
**Acceptance:**
- `pytest tests/test_render_cli.py -v` passes (config parsing + scene wiring; the
  full bpy render path may be a bpy-marked smoke test that renders 1–2 frames from a
  tiny config). Frames appear under `output/`. `ruff` + `mypy` clean.

---

### T9 — Frontend scaffold (`frontend/`)
**Goal:** Vite + React + TS + Three.js project wired to the API in dev.
**Depends on:** none (consumes T6 at runtime).
**Files:** create `frontend/package.json`, `frontend/vite.config.ts`,
`frontend/tsconfig.json`, `frontend/index.html`, `frontend/src/main.tsx`,
`frontend/src/App.tsx`.
**Steps:**
- Scaffold a Vite React-TS app. Add `three`. Use **yarn**.
- Apply the **`ct-frontend-design`** skill for component structure and styling
  conventions.
- `vite.config.ts`: dev-proxy `/api` → `http://127.0.0.1:8000`.
**Acceptance:**
- `yarn install` + `yarn build` succeed; `yarn dev` serves a blank app that can
  reach `/api/health` through the proxy. Type-check (`tsc --noEmit`) clean.

---

### T10 — Typed API client (`frontend/src/api.ts`)
**Goal:** One typed place for backend calls; debounced.
**Depends on:** T6, T9.
**Files:** create `frontend/src/api.ts`, `frontend/src/types.ts`.
**Steps:**
- Typed wrappers: `fetchMesh(path)`, `fetchPoses(params)`, `saveConfig(config, filename)`.
- Debounce `fetchPoses` (~100 ms) for slider drags. Types mirror the pydantic models.
**Acceptance:** type-check clean; calls hit the proxied endpoints and return typed data.

---

### T11 — Three.js preview component (`frontend/src/preview/Viewport.tsx`)
**Goal:** The 3D view. **Display only — no sampling/geometry math here.**
**Depends on:** T10.
**Files:** create `frontend/src/preview/Viewport.tsx` (+ helpers).
**Steps:**
- Scene with `OrbitControls`, basic lighting, axes/grid.
- Object: build a `BufferGeometry` from `fetchMesh` positions/indices; flat grey
  material. The mesh arrives **already oriented + centered** — apply no rotation.
- Band: surface + edge lines from `fetchPoses`. Camera points: small markers
  (toggleable). Look-at rays: optional thin lines.
- Ground plane from the response (semi-transparent), so a band dipping below the
  object's base is obvious.
**Acceptance:** loading the sample `.obj` shows the grey mesh wrapped by a band that
visibly moves when elevation/azimuth change; the mesh orientation matches what the
render produces (cross-check against a `rembrandt render` frame). Type-check clean.

---

### T12 — Controls panel (`frontend/src/controls/`)
**Goal:** Path input + camera-sphere sliders driving the preview.
**Depends on:** T11.
**Files:** create `frontend/src/controls/Controls.tsx` (+ inputs).
**Steps:**
- Path text input (paste a disk path) → triggers `fetchMesh`.
- Controls for `n`, `azimuth_range`, `elevation_range`, `distance_range`, `strategy`,
  `seed` → trigger debounced `fetchPoses`. Follow `ct-frontend-design` patterns.
- Hold the full config in React state (camera params editable; lights/render/output
  carry sensible defaults, editable later).
**Acceptance:** changing any control updates the preview live; values stay within the
sampler's valid ranges. Type-check clean.

---

### T13 — Save config UI
**Goal:** Persist the dialed-in config.
**Depends on:** T12, T6.
**Files:** edit `frontend/src/App.tsx` / add a `SaveBar` component.
**Steps:**
- A filename field + **Save** button → `saveConfig`. Show the written path on success
  and validation errors on failure. Optionally render a read-only YAML preview of the
  current config.
**Acceptance:** Save writes `./configs/<name>.yaml`; the file is consumable by
`rembrandt render` (manual end-to-end). Type-check clean.

---

### T14 — Packaging, scripts, dev ergonomics, README
**Goal:** "Clone from GitHub and run." No PyPI.
**Depends on:** T7, T8, T9.
**Files:** edit `pyproject.toml`, `README.md`; add `Makefile` (or root `package.json`
script).
**Steps:**
- Finalize `[project.dependencies]` (fastapi, uvicorn, pydantic, pyyaml, typer, rich)
  and `[project.scripts]` (`rembrandt-serve`, `rembrandt-render`).
- Add a `make dev` (or `concurrently`) target running Vite dev server + `uvicorn --reload`.
- README **Setup** section: the bpy prerequisites are the real friction — document
  the apt system libs (`libxrender1`, `libxi6`, `libxxf86vm1`, `libxfixes3`,
  `libxkbcommon0`, `libsm6`, `libgl1`), Python 3.11, then
  `pip install -e ".[dev]"`, `cd frontend && yarn install && yarn build`,
  then `rembrandt-serve`. Document `make dev` for development.
**Acceptance:** a fresh clone, following the README, reaches a working
`rembrandt-serve` (built SPA served, preview works) and `rembrandt-render cfg.yaml`
(frames written).

---

### T15 — Retire the Streamlit/Plotly preview
**Goal:** Remove the second, now-superseded preview so the two can't drift.
**Depends on:** T11–T13 working.
**Files:** delete `scripts/camera_pose_preview.py`, `.streamlit/config.toml`; edit
`pyproject.toml` (remove the `preview` extra: `plotly`, `streamlit`); edit `README.md`
(replace the "camera pose preview" section with the SPA instructions).
**Steps:** confirm all reusable logic was ported in T3/T4 before deleting; remove the
now-dead extra and docs.
**Acceptance:** no remaining import of streamlit/plotly; `ruff` clean; README points
only at the SPA.

---

## 4. Definition of done
- `pip install -e ".[dev]"` then `pytest -v` is green (bpy-marked tests run in the bpy
  env; pure tests run without it). `ruff check`, `ruff format --check`, and `mypy`
  (strict) clean across `src/` and `tests/`.
- `rembrandt-serve` → open browser → paste the `test-obj/` path → grey mesh appears
  wrapped by the angle band → dragging elevation moves the band relative to the object
  and the ground plane → **Save** writes `./configs/<name>.yaml`.
- `rembrandt-render ./configs/<name>.yaml` renders frames whose object orientation
  matches the preview.
- The web server never imports bpy; the orientation/centering convention exists in
  exactly one module; no pose or geometry math is duplicated in the frontend.