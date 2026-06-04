# Rembrandt — Randomized Backgrounds (Post-Composite): Implementation Plan

> **For the implementing agent (Cursor Composer):** Read this whole file first, then
> `.ai/AGENTS.md`, `src/rembrandt/config.py`, `src/rembrandt/render.py`,
> `src/rembrandt/scene.py`, `src/rembrandt/camera_poses.py` (style reference for
> pure modules and seeding discipline), and `src/rembrandt/errors.py`. Execute the
> tasks **in order** — each lists its dependencies, files, steps, and acceptance
> criteria. Do not start a task until its dependencies pass their acceptance gates.

---

## 1. Overview

### What we're building

Rendered frames currently show the object against Blender's default world
background. We are adding **randomized real-photo backgrounds** using a
**post-composite** approach, with **BG-20k** (20,000 high-resolution
background photos *without salient objects*, MIT-licensed, mirrored on the
Hugging Face Hub) as the recommended source:

1. `Scene.render` gains an opt-in transparent-film mode that outputs **RGBA**
   frames with the object cut out against alpha.
2. A new **bpy-free** module `src/rembrandt/backgrounds.py` indexes a directory
   of background images, deterministically picks one per frame, resizes it to
   cover the render resolution, and alpha-composites the rendered foreground
   over it.
3. `render_from_config` wires the two together when the config enables it,
   overwriting each `frame_XXXX.png` with the composited RGB result.
4. A new **`rembrandt-fetch-backgrounds`** command (optional `[backgrounds]`
   extra) materializes a bounded BG-20k sample from the Hugging Face Hub into
   a local directory **once, ahead of time**. The render pipeline itself stays
   fully offline and only ever reads `background.image_dir`.

BG-20k is preferred over Open Images because it was built specifically as
compositing backgrounds: it contains no salient objects, so it cannot create
unlabeled positives ("label leakage") once YOLO labels exist.

### Why post-composite (and not an in-Blender background)

The background here is **distractor content for a detection dataset**, not a
scene we are trying to make physically plausible. Compositing in post:

- keeps the bpy surface change to two render settings in `scene.py`;
- puts all interesting logic (indexing, seeded selection, resize/crop, blend)
  in a pure-Python module testable in the fast lane, matching the project's
  bpy-free philosophy;
- guarantees the object's **pixel position is untouched**, which future YOLO
  label generation depends on;
- slots into the `backgrounds.py` module already named on the `.ai/AGENTS.md`
  roadmap.

### Out of scope (do **not** build here)

- In-Blender compositor node setups, environment textures / HDRI domes,
  image-based lighting, or shadow catchers. (Those are a possible later
  "realism" path and are bpy-heavy by nature.)
- Network access anywhere in the render or preview pipelines. Fetching
  happens only in the dedicated `rembrandt-fetch-backgrounds` command (T5);
  `rembrandt-render` and `rembrandt-serve` never touch the network and the
  user may still point `background.image_dir` at any hand-supplied directory.
- Any dataset source beyond the one default BG-20k Hub repo in the fetch
  command (no FiftyOne/Open Images integration, no multi-source plugin
  system). Alternatives are documented in the README only.
- Any 2D augmentation beyond the alpha-over composite (no blur, color jitter,
  random placement, scaling, or copy-paste of the foreground).
- YOLO labels, bbox projection, train/val split — unchanged roadmap items.
- SPA UI controls for backgrounds. The preview is about camera angles;
  backgrounds have no preview meaning. (A minimal type mirror in the frontend
  is included in T6 only so `createDefaultConfig` stays an honest mirror of
  `RembrandtConfig`.)
- Per-frame background-provenance metadata files. Stdout logging is enough.

---

## 2. Guardrails (apply to every task)

1. **The bpy-free boundary is unchanged.** `backgrounds.py`, `config.py`, and
   everything under `web/` and `preview/` must not `import bpy` or import any
   module that does. bpy stays only in `scene.py`, `camera/orientation.py`
   (lazy), and behind `Scene` for `render.py`. Add a
   `test_backgrounds_module_is_bpy_free` source-scan test mirroring the
   existing pattern (`test_web_api_module_is_bpy_free`).
2. **`render.py` must keep passing
   `test_render_module_only_imports_bpy_through_scene`.** It may import
   `backgrounds` (pure) but never bpy.
3. **The foreground must not move.** The composite resizes/crops the
   *background* to the foreground's exact resolution. Never scale, translate,
   pad, or crop the rendered foreground — future bounding boxes must line up
   with pixels exactly.
4. **Determinism follows the existing seeding discipline.** Background choice
   uses a local `random.Random` derived from `background.seed` plus the frame
   index. Same config → byte-identical background choices. Never touch the
   global RNG state (there is a test for this pattern in
   `test_camera_poses.py` — replicate it).
5. **Existing behavior is the default.** `background.mode` defaults to
   `"none"`; existing YAML configs, the SPA-saved configs, and all current
   tests must load and behave unchanged. `Scene.render` keeps its current RGB
   output unless transparent film is explicitly requested.
6. **Fail fast, before rendering.** If the config enables image backgrounds
   but the directory is missing or contains no usable images, raise before the
   first frame renders — not after a long render loop.
7. **Match repo conventions.** `from __future__ import annotations`;
   Google-style docstrings (`Args:` / `Returns:` / `Raises:`); `Literal` for
   enum-like params; keyword-only args where the surrounding code uses them.
   Tooling is **pip + hatchling** — do not introduce UV/Docker. `ruff check`,
   `ruff format --check`, and `mypy --no-sqlite-cache src` (strict) must stay
   clean. New dependency `pillow` goes in `[project.dependencies]`
   (`ignore_missing_imports = true` already covers any stub gaps).
8. **Network and heavy deps are quarantined in the fetch command.** The
   `datasets` library goes in a new `[project.optional-dependencies]`
   `backgrounds` extra, **never** in core dependencies. Only
   `fetch_backgrounds.py` may import it, and only **lazily** (inside the
   function, like the `bpy` pattern in `camera/orientation.py`), with a
   friendly error telling the user to run `pip install -e ".[backgrounds]"`
   when it is missing. `rembrandt-render`, `rembrandt-serve`, and all of
   `backgrounds.py` remain importable and fully functional without the extra
   installed — CI's existing lanes must not need it.

---

## 3. Atomic tasks

### T1 — Config schema: `BackgroundConfig`

**Goal:** The YAML contract for backgrounds, default-off.
**Depends on:** none.
**Files:** edit `src/rembrandt/config.py`, `tests/test_config.py`.

**Steps:**

- Add to `config.py`:

  ```python
  class BackgroundConfig(BaseModel):
      """Randomized background compositing (post-render). Off by default."""

      mode: Literal["none", "image"] = "none"
      image_dir: str | None = None
      seed: int | None = None

      @model_validator(mode="after")
      def _check_image_dir(self) -> Self:
          if self.mode == "image" and not self.image_dir:
              raise ValueError("background.image_dir is required when mode is 'image'")
          return self
  ```

- Add `background: BackgroundConfig = Field(default_factory=BackgroundConfig)`
  to `RembrandtConfig`.
- `image_dir` path semantics mirror `object.path`: absolute, relative to the
  config file, or relative to the CWD (resolution itself happens in T4).
- Document in the model docstring that `background.seed` is independent of
  `camera.seed`; `seed: null` means non-reproducible background choice.

**Acceptance:**

- `pytest tests/test_config.py -v` passes with new cases:
  - defaults: a config without a `background` block loads with
    `mode == "none"`, `image_dir is None`, `seed is None`;
  - round-trip `dump_config` → `load_config` equality including a populated
    `background` block;
  - `mode: image` without `image_dir` raises `ValidationError`
    matching `image_dir`;
  - `mode: dome` (or any other value) rejected.
- Existing config tests pass unchanged. `ruff` + `mypy` clean. No bpy import.

---

### T2 — Pure background module: `backgrounds.py`

**Goal:** All background logic as a bpy-free, fully unit-tested module.
**Depends on:** none (T1 only meets it in T4).
**Files:** create `src/rembrandt/backgrounds.py`, `tests/test_backgrounds.py`;
edit `src/rembrandt/errors.py`, `pyproject.toml` (add `pillow`).

**Steps:**

- Add to `errors.py`:

  ```python
  class BackgroundDirectoryNotFoundError(FileNotFoundError):
      """Raised when a background image directory cannot be found."""

      def __init__(self, dir_path: str) -> None:
          self.dir_path = dir_path
          super().__init__(f"Background directory not found: {dir_path}")
  ```

- Implement in `backgrounds.py` (module docstring must state it is bpy-free
  and why — same spirit as `camera_poses.py`):

  - `_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}` (module constant).
  - `index_backgrounds(directory: str | Path) -> list[Path]` — recursive
    (`rglob`), case-insensitive extension match, **sorted** result for
    determinism. Raises `BackgroundDirectoryNotFoundError` if the directory
    does not exist; raises `ValueError` (`"no background images found in ..."`)
    if the pool is empty.
  - `choose_background(backgrounds: list[Path], *, frame_index: int, seed: int | None) -> Path`
    — uses `Random(seed + frame_index)` when `seed is not None`, else a fresh
    unseeded local `Random()`. Raises `ValueError` on an empty list or
    `frame_index < 0`. Must not mutate global RNG state.
  - `load_cover_resized(path: str | Path, *, width: int, height: int) -> npt.NDArray[np.uint8]`
    — open with Pillow, `convert("RGB")`, scale so the image **covers**
    `width × height` (scale factor `max(width / w, height / h)`), then
    center-crop to exactly `(height, width, 3)`. Use a high-quality resample
    filter (LANCZOS).
  - `composite_over(foreground_rgba: npt.NDArray[np.uint8], background_rgb: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]`
    — pure numpy alpha-over: validate foreground shape `(h, w, 4)` and
    background shape `(h, w, 3)` with matching `h, w` (raise `ValueError`
    otherwise); blend in float64
    (`out = fg_rgb * a + bg_rgb * (1 - a)`), round, return uint8 RGB. No
    geometric transforms of the foreground.
  - `apply_background_to_frame(frame_path: str | Path, background_path: str | Path) -> Path`
    — convenience used by the render loop: read the frame as RGBA
    (`Image.open(...).convert("RGBA")`), `load_cover_resized` the background to
    the frame's size, `composite_over`, write the RGB result back to
    `frame_path` (overwrite, PNG), return the path.

**Acceptance:**

- `pytest tests/test_backgrounds.py -v` passes, covering at minimum:
  - indexing: finds nested files, ignores other extensions, sorted output,
    missing dir raises `BackgroundDirectoryNotFoundError`, empty dir raises
    `ValueError`;
  - choice: same `(seed, frame_index)` → same path; different frame indices
    with the same seed vary across a multi-image pool; `seed=None` works;
    global-RNG-non-mutation test mirroring
    `test_sample_camera_poses_does_not_mutate_global_rng`;
  - cover-resize: output shape exactly `(height, width, 3)` for wider-than,
    taller-than, and exact-aspect inputs (build tiny test images with Pillow
    in `tmp_path` — no committed binary fixtures needed);
  - composite math on tiny synthetic arrays: alpha=255 pixel → foreground
    value; alpha=0 → background value; alpha=128 → blended value within ±1;
    shape mismatch raises `ValueError`;
  - `apply_background_to_frame` round-trip: write a small RGBA PNG with a
    transparent region, apply, reload, assert mode is RGB and the transparent
    region now shows background pixels;
  - `test_backgrounds_module_is_bpy_free` source scan.
- All of the above run in the fast lane (`pytest -m "not bpy" -q`).
- `ruff` + `mypy` clean. `pillow` added to `[project.dependencies]`.

---

### T3 — Transparent-film render mode in `Scene.render`

**Goal:** Opt-in RGBA output with the object cut out against alpha. The only
bpy change in this plan.
**Depends on:** none.
**Files:** edit `src/rembrandt/scene.py`, `tests/test_render_cli.py` or a new
bpy-marked test file.

**Steps:**

- Add a keyword-only parameter to `Scene.render`:
  `transparent_film: bool = False`.
- When `True`: set `bpy_scene.render.film_transparent = True` and
  `image_settings.color_mode = "RGBA"`.
- When `False`: **explicitly** set `film_transparent = False` and
  `color_mode = "RGB"` (do not rely on prior state — bpy scene settings are
  global and a previous render in the same process may have flipped them).
- Update the `render` docstring (`Args:` entry) noting this is consumed by the
  background compositing step. Works for both `EEVEE` (EEVEE Next) and
  `CYCLES`.

**Acceptance:**

- A new bpy-marked smoke test (pattern of `test_render_smoke_writes_frames`):
  render one 64×64 frame of a sample object with `transparent_film=True`,
  reload it with Pillow, assert the image has an alpha channel and that at
  least one corner pixel has `alpha == 0` while the center region contains
  `alpha > 0` pixels (the object is centered, corners are empty for the
  fixture assets). Skips cleanly without bpy; fails loudly under
  `--require-bpy`.
- A second bpy-marked case (or the existing smoke test) confirms the default
  path still writes an RGB (no-alpha) image, proving the explicit reset.
- `pytest -m bpy --require-bpy -q` green; `ruff` + `mypy` clean.

---

### T4 — Wire compositing into `render_from_config`

**Goal:** Config-driven backgrounds in the render loop, failing fast.
**Depends on:** T1, T2, T3.
**Files:** edit `src/rembrandt/render.py`, `tests/test_render_cli.py`.

**Steps:**

- Add `resolve_background_dir(config_path: Path, image_dir: str) -> Path`
  alongside `resolve_object_path`, with the same resolution order: absolute →
  relative to the config file → relative to the CWD (directory existence
  checked via `is_dir()`).
- In `render_from_config`, **before the render loop**:

  ```python
  use_background = cfg.background.mode == "image"
  background_pool: list[Path] = []
  if use_background:
      assert cfg.background.image_dir is not None  # guaranteed by T1 validator
      bg_dir = resolve_background_dir(config_path, cfg.background.image_dir)
      background_pool = index_backgrounds(bg_dir)  # raises before any frame renders
  ```

- Pass `transparent_film=use_background` to every `scene.render(...)` call.
- After each frame is rendered, when `use_background`:

  ```python
  background_path = choose_background(
      background_pool, frame_index=index, seed=cfg.background.seed
  )
  apply_background_to_frame(frame_path, background_path)
  ```

  Extend the per-frame stdout line to include the chosen background filename.
- Output layout is unchanged: the composited RGB result overwrites
  `frame_XXXX.png` in place.

**Acceptance:**

- `pytest tests/test_render_cli.py -v` passes with new cases (all bpy-free,
  using the existing `MagicMock` scene-factory pattern):
  - `mode: none` → `scene.render` called with `transparent_film=False`; no
    background module interaction; existing wiring assertions still pass
    (update `scene.render.assert_any_call` expectations for the new kwarg).
  - `mode: image` with a `tmp_path` background dir containing 2–3 tiny PNGs
    (written with Pillow in the test) → `scene.render` called with
    `transparent_film=True`; have the mock's `render.side_effect` write a
    small real RGBA PNG (with a transparent region) to the requested path;
    after `render_from_config` returns, reload each frame and assert it is
    RGB (alpha gone) and the previously transparent region matches background
    pixels.
  - determinism: two runs with the same `background.seed` pick identical
    background sequences (assert via monkeypatched `choose_background`
    capture or by using visually distinct solid-color backgrounds and
    comparing composited corner pixels).
  - fail-fast: `mode: image` pointing at a missing dir raises
    `BackgroundDirectoryNotFoundError` and `scene.render` was **never**
    called; an existing-but-empty dir raises `ValueError`, same guarantee.
- `test_render_module_only_imports_bpy_through_scene` still passes.
- Optionally extend the bpy smoke test to render 2 frames with `mode: image`
  end-to-end and assert the outputs are RGB.
- `ruff` + `mypy` clean.

---

### T5 — `rembrandt-fetch-backgrounds` (BG-20k fetch command)

**Goal:** A one-time, opt-in command that materializes a bounded BG-20k
sample from the Hugging Face Hub into a local directory consumable by
`background.image_dir`. This is the **only** network-aware code in the
project.
**Depends on:** T2 (reuses `_EXTENSIONS` conventions; output must satisfy
`index_backgrounds`).
**Files:** create `src/rembrandt/fetch_backgrounds.py`,
`tests/test_fetch_backgrounds.py`; edit `pyproject.toml`.

**Steps:**

- `pyproject.toml`:
  - add `[project.optional-dependencies]` entry: `backgrounds = ["datasets"]`;
  - register `[project.scripts]`
    `rembrandt-fetch-backgrounds = "rembrandt.fetch_backgrounds:main"`.
- Implement a typer app mirroring the structure of `render.py` (typer `app`,
  `main()` entry, Google-style docstrings):

  ```python
  DEFAULT_DATASET = "unography/BG-20k-1200px"

  @app.command()
  def fetch_command(
      out_dir: Annotated[Path, typer.Option("--out", help="Destination directory.")] = Path("backgrounds"),
      count: Annotated[int, typer.Option(min=1, help="Number of images to fetch.")] = 2000,
      dataset: Annotated[str, typer.Option(help="Hugging Face dataset repo id.")] = DEFAULT_DATASET,
      split: Annotated[str, typer.Option(help="Dataset split to stream from.")] = "train",
  ) -> None: ...
  ```

- Structure the module as two layers so the network edge is thin and the rest
  is unit-testable without `datasets`:
  - `_stream_dataset_images(dataset: str, split: str) -> Iterator[Image.Image]`
    — the **only** function that imports `datasets`, lazily inside the
    function body. On `ImportError`, raise a clear error:
    `"the 'datasets' package is required: pip install -e \".[backgrounds]\""`.
    Use `load_dataset(dataset, split=split, streaming=True)` and yield each
    sample's `image` field (streaming avoids downloading the full 20k-row
    parquet set when `count` is small).
  - `write_background_images(images: Iterable[Image.Image], *, out_dir: Path, count: int) -> list[Path]`
    — pure writer: create `out_dir` (parents ok), iterate `islice(images,
    count)`, `convert("RGB")`, save as `out_dir / f"bg_{i:05d}.jpg"` with
    `quality=90`, return the written paths. Warn (stdout) if the stream
    yielded fewer than `count` images.
- The command body is just: stream → write → echo a summary line with the
  directory, image count, and a reminder to set
  `background.image_dir: <out_dir>`.
- Print the BG-20k attribution line (MIT license; "Bridging Composite and
  Real: Towards End-to-End Deep Image Matting", IJCV 2021) in the summary so
  the provenance travels with the fetch.

**Acceptance:**

- `pytest tests/test_fetch_backgrounds.py -v` passes **without** the
  `backgrounds` extra installed, covering:
  - `write_background_images` with a generator of tiny in-memory PIL images:
    correct filenames (`bg_00000.jpg`…), exact count, RGB mode on reload,
    short-stream case writes fewer and warns;
  - the written directory is directly consumable: `index_backgrounds(out_dir)`
    (T2) returns exactly the written files in order;
  - CLI wiring via `typer.testing.CliRunner` with
    `_stream_dataset_images` monkeypatched to a fake generator — no network,
    no `datasets` import;
  - the lazy-import error message: monkeypatch the import to raise
    `ImportError` and assert the `pip install -e ".[backgrounds]"` hint
    surfaces;
  - `test_fetch_backgrounds_module_is_bpy_free` source scan (and no
    **top-level** `datasets`/network import — assert `"from datasets"` /
    `"import datasets"` only appears inside the lazy function, mirroring how
    `camera/orientation.py` handles bpy).
- `pip install -e ".[dev]"` alone (no extra) still imports the module and runs
  the full fast lane green — the extra is needed only to actually fetch.
- Manual check (network, not CI):
  `rembrandt-fetch-backgrounds --out ./backgrounds --count 50` produces 50
  JPEGs that a `mode: image` render consumes end-to-end.
- `ruff` + `mypy` clean.

---

### T6 — Docs, sample config, frontend type mirror

**Goal:** Discoverability, and keep the SPA's default-config mirror honest.
**Depends on:** T1–T5.
**Files:** edit `README.md`, `configs/dataset.yaml` (or add a commented
example), `frontend/src/types.ts`, `frontend/src/defaultConfig.ts`.

**Steps:**

- README:
  - extend the **Config Format** example with the `background` block
    (commented as optional, default `none`);
  - add a short **Randomized backgrounds** subsection: how it works
    (transparent film + post-composite), determinism via `background.seed`,
    and the recommended workflow:

    ```bash
    pip install -e ".[backgrounds]"
    rembrandt-fetch-backgrounds --out ./backgrounds --count 2000
    # then in your config:
    #   background:
    #     mode: image
    #     image_dir: ./backgrounds
    #     seed: 7
    ```

  - explain **why BG-20k**: purpose-built compositing backgrounds with no
    salient objects, so no risk of the rendered object's class appearing
    unlabeled in a background once YOLO labels exist; MIT license — cite
    "Bridging Composite and Real: Towards End-to-End Deep Image Matting"
    (IJCV 2021) if the generated dataset is redistributed;
  - document the alternative: any local image directory works (the fetch
    command is just a convenience). If using **Open Images** instead,
    download a bounded subset via the FiftyOne zoo loader
    (`foz.load_zoo_dataset("open-images-v7", split="validation",
    max_samples=N)`) — never the full ~9M set — and include the
    **label-leakage warning**: if the rendered object's class appears in
    background photos, filter the pool by Open Images class labels; Open
    Images annotations are CC-BY and the photos are Flickr images under
    CC-BY-style licenses, so attribution applies on redistribution.
- Frontend (no UI controls, types only): add to `types.ts`

  ```ts
  export type BackgroundMode = "none" | "image";

  export type BackgroundConfig = {
    mode?: BackgroundMode;
    image_dir?: string | null;
    seed?: number | null;
  };
  ```

  add `background?: BackgroundConfig` to `RembrandtConfig`, and include
  `background: { mode: "none" }` in `createDefaultConfig` so the comment
  "Defaults mirroring `rembrandt.config.RembrandtConfig`" stays true.

**Acceptance:**

- `cd frontend && yarn typecheck && yarn build` clean.
- Saving a config from the SPA still round-trips through
  `POST /api/config/save` and `load_config` (existing `test_save_config_writes_yaml`
  passes; pydantic fills the background defaults).
- README renders the new section; the example YAML in the README validates via
  `load_config`.

---

## 4. Definition of done

- `pip install -e ".[backgrounds]"` followed by
  `rembrandt-fetch-backgrounds --out ./backgrounds --count 2000` fills a local
  directory with BG-20k JPEGs, and a config with

  ```yaml
  background:
    mode: image
    image_dir: ./backgrounds
    seed: 7
  ```

  renders frames where the object appears composited over varied real photos;
  rerunning the same config reproduces the identical background sequence.
- A config without a `background` block (every existing config) behaves
  exactly as today: RGB frames, default world background, no new code paths
  executed.
- The render and preview pipelines never touch the network; `datasets` is
  imported lazily and only by `fetch_backgrounds.py`; everything except the
  actual fetch works without the `[backgrounds]` extra installed.
- All background logic lives in bpy-free `backgrounds.py`; `scene.py` gained
  only the `transparent_film` switch; `render.py` still imports bpy only
  through `Scene`.
- The foreground's pixel geometry is provably untouched (composite tests
  assert pixel-exact foreground values where alpha is opaque).
- `pytest -m "not bpy" -q`, `pytest -m bpy --require-bpy -q`, `ruff check src
  tests`, `ruff format --check src tests`, and `mypy --no-sqlite-cache src`
  are all clean — without the `backgrounds` extra installed.

## 5. Quick verification commands

```bash
pytest -m "not bpy" -q                       # fast lane incl. backgrounds + fetch tests (no extra needed)
pytest -m bpy --require-bpy -q               # transparent-film smoke + render smoke
ruff check src tests && ruff format --check src tests
mypy --no-sqlite-cache src
cd frontend && yarn typecheck && yarn build

# one-time fetch (network; requires the extra)
pip install -e ".[backgrounds]"
rembrandt-fetch-backgrounds --out ./backgrounds --count 50

# end-to-end eyeball
rembrandt-render ./configs/dataset-with-backgrounds.yaml
# inspect output/<stamp>/frame_0000.png — object over a photo, no alpha channel
```