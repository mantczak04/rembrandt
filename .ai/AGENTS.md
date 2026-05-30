# CLAUDE.md

Guidance for working in the **Rembrandt** repository. Read this before making changes.

## What this project is

Rembrandt generates synthetic computer-vision training datasets from 3D models. The
user supplies an `.obj`, configures camera coverage and lighting, and Rembrandt renders
many images of the object from randomized camera poses.

It is **early-stage**. Today the render step writes PNG frames only. YOLO labels,
train/val dataset layout, 2D augmentations, and training-script generation are planned
follow-up work — do not assume they exist.

## Two tools, one shared config

The project is two cooperating but independent tools that communicate through a single
YAML config (`RembrandtConfig` in `src/rembrandt/config.py`):

- **`rembrandt-serve`** — a bpy-free FastAPI server + React/Three.js SPA for choosing an
  `.obj`, previewing camera-angle coverage in 3D, and saving a config. It is a
  *configuration-sanity tool*: it shows the sampled camera band relative to the oriented
  object and ground plane so the user can judge whether the angular coverage makes sense.
  It does **not** preview rendered images, trigger renders, or monitor progress.
- **`rembrandt-render CONFIG_PATH`** — the Blender/`bpy` render step that reads the YAML
  and writes frames.

## Hard rules (do not break these)

These are the load-bearing invariants. Several have tests that will fail loudly if
violated, but treat them as design constraints, not just test targets.

1. **The web/preview stack is bpy-free.** Nothing under `src/rembrandt/web/`,
   `src/rembrandt/preview/`, nor `config.py` or `convention.py` may `import bpy` or import
   any module that does. There are `*_module_is_bpy_free` tests asserting this by scanning
   source for `import bpy` / `from bpy`. bpy is allowed **only** in:
   - `scene.py` (imports `bpy` directly)
   - `camera/orientation.py` (lazy — imports `bpy`/`mathutils` *inside* functions)
   - `render.py` orchestrates the bpy pipeline **through `Scene`** but does not itself
     `import bpy` (see `test_render_module_only_imports_bpy_through_scene`).
2. **Python is the single source of truth; the frontend only displays.** The SPA renders
   numbers the backend hands it. Never port pose sampling, band geometry, or the
   orient/center transform into TypeScript. If the frontend needs a computed value, add
   an endpoint.
3. **One orientation/centering convention, defined once.** `convention.py` holds both the
   `bpy.ops.wm.obj_import` axes and the pure-Python `orient_and_center()` mapping used by
   the preview. Source OBJ up-axis is explicit (`object.up_axis`, default `Y`, optional
   `Z`) and both paths must derive from that declaration — they must not drift.
   `test_orient_and_center_matches_bpy_import` (bpy-marked) is the parity check.
4. **+Z is world up; cameras/lights use -Z forward, Y up.** Elevation is measured from the
   XY plane (`z = distance * sin(elevation)`). Objects are oriented from their declared
   source up-axis into +Z and centered on their bounding-box center.
5. **The preview is about angles, not distance.** Band/ground-plane radius is cosmetic
   (a display radius so the band wraps the object legibly). Do **not** apply camera-fit /
   framing math in the preview — fit only affects distance, which the preview ignores.
6. **Single object input source.** The user pastes a filesystem path; there is no upload.

## Commands

Python 3.11 is required (see "Gotchas").

```bash
# Backend (editable install with dev tooling)
pip install -e ".[dev]"

# Frontend build (Yarn; enable via `corepack enable` if needed)
cd frontend && yarn install && yarn build      # or: make build-frontend

# Run both servers together for development
make dev          # uvicorn --reload on :8000  +  Vite dev server on :5173 (proxies /api)

# Launch the configurator
rembrandt-serve                                # FastAPI + SPA at http://127.0.0.1:8000/

# Render frames from a config
rembrandt-render ./configs/dataset.yaml        # writes <output.dir>/<timestamp>/frame_XXXX.png
```

### Checks before considering work done

```bash
pytest -m "not bpy" -q          # fast bpy-free lane (default local iteration)
pytest -m bpy --require-bpy -q  # orientation parity + render smoke (needs bpy)
ruff check src tests
ruff format --check src tests
mypy --no-sqlite-cache src
cd frontend && yarn typecheck && yarn build
```

CI runs both lanes (see `.github/workflows/ci.yml`). The bpy job passes
`--require-bpy` so parity tests cannot pass-by-skipping when Blender is expected.
Committed orientation fixtures live in `tests/fixtures/`; optional full `.obj` files
may also be placed in `test-obj/` (gitignored).

All of these must be clean. `ruff` is configured with line-length 100 and rule sets
`E,W,F,I,B,UP,N`; `mypy` runs in **strict** mode (`ignore_missing_imports = true` because
bpy ships no type stubs).

## Code conventions

Match the existing style in `scene.py` and `camera_poses.py`:

- `from __future__ import annotations` at the top of every module.
- Google-style docstrings with `Args:` / `Returns:` / `Raises:` sections.
- `Literal[...]` for enum-like params (e.g. `SamplingStrategy`, `RenderEngine`).
- Keyword-only arguments where the surrounding code uses them.
- Domain failures get custom exceptions (`errors.py`, e.g. `ModelFileNotFoundError`);
  generic validation uses stdlib `ValueError`. API routes map these to 4xx.
- Every new pure module gets tests that run **without** bpy.
- Tooling is **pip + hatchling**, not UV. Do not introduce UV/Docker/k8s here — those are
  org defaults that this project does not use.
- Frontend work follows the **`ct-frontend-design`** skill for component structure and
  styling; the frontend uses React 19 + TypeScript + Three.js + Vite, managed with Yarn.

## Module map (current, actual state)

```
src/rembrandt/
├── camera_poses.py        # pure pose sampling (random / fibonacci). NO bpy.
├── config.py              # pydantic v2 schema + YAML load/dump. NO bpy.
├── convention.py          # axis constants + orient_and_center(). Single source of truth. NO bpy.
├── errors.py              # ModelFileNotFoundError, etc.
├── render.py              # `rembrandt-render` entry; drives Scene. (no direct bpy import)
├── scene.py               # Blender scene wrapper. bpy lives here.
├── camera/
│   ├── fit.py             # pure fit_distance() math. NO bpy.
│   ├── intrinsics.py      # K-matrix / FOV math (BlenderProc port). NO bpy.
│   └── orientation.py     # forward-vec -> euler. Lazy bpy/mathutils import only.
├── preview/
│   ├── mesh.py            # bpy-free .obj parse -> oriented PreviewMesh. NO bpy.
│   └── geometry.py        # bpy-free band/points/ground-plane builders. NO bpy.
└── web/
    ├── app.py             # FastAPI factory + SPA static serving. NO bpy.
    ├── api.py             # /preview/mesh, /preview/poses, /config/save. NO bpy.
    └── serve.py           # `rembrandt-serve` uvicorn entry. NO bpy.
```

Entry points (`pyproject.toml`): `rembrandt-serve = rembrandt.web.serve:main`,
`rembrandt-render = rembrandt.render:main`. `main.py` is a legacy shim that just calls
`render.main`.

The planned-but-not-yet-built modules listed in `.ai/AGENTS.md` (`randomize.py`,
`annotations.py`, `augment.py`, `backgrounds.py`, `dataset.py`, `templates/`) do not
exist yet — that tree describes the intended layout, not the current one.

## Testing notes

- Pure-module tests run without bpy and are the fast majority (`pytest -m "not bpy"`).
- Tests that need the Blender runtime are marked `@pytest.mark.bpy` and/or use
  `pytest.importorskip("bpy")`, so they skip cleanly in a bpy-less environment.
- The bpy CI lane runs `pytest -m bpy --require-bpy` so missing bpy is a hard failure.
  Key guards: `test_orient_and_center_matches_bpy_import`,
  `test_preview_mesh_matches_scene_geometry` (see `tests/test_orientation_parity.py`).
- The bpy-free guarantee is enforced by source-scanning tests (e.g.
  `test_web_api_module_is_bpy_free`). If you add a bpy import to a forbidden module, that
  test — not a runtime crash — is what catches it.
- Web routes are tested via `fastapi.testclient.TestClient`.

## Gotchas

- **bpy is pinned 1:1 to Python 3.11.** The `bpy` wheel ties to a specific Python minor
  version. Bumping Python means bumping bpy and vice versa. `.python-version` pins 3.11.
- **The `bpy` wheel is ~700MB.** CI installs need aggressive caching.
- **`src/` layout:** you must `pip install -e .` before `import rembrandt` works; running
  scripts from the repo root won't find the package otherwise.
- **Camera fit can override sampled distances.** `Scene.add_camera` and `Scene.move_camera`
  default to `fit_target=True`: if a sampled distance is too close to frame the object,
  the camera is pushed back to the fit distance. For large objects with a small
  `distance_range`, many sampled distances may be overridden. Pass `fit_target=False` for
  exact sampled distances.
- **`output.train_val_split` is reserved** and not consumed by the current frame renderer.
- **Object paths in YAML** may be absolute, relative to the config file, or relative to the
  CWD — `resolve_object_path` tries them in that order.

## Further context

The `.ai/specs/` directory holds the design specs that produced the current code, including the SPA-pivot
plan (`spa-migration.-29-05.md`) and the camera/scene refactor spec — read the relevant
spec before extending those areas.