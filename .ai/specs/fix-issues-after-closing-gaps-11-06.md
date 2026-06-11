# Rembrandt — Post-Review Fixes: Implementation Plan

> Scope: fix the issues found in the review of the `close-gaps-ultraplan-11-06` implementation.
> Five fix tracks, ordered by severity: (A) parallel workers run sequentially, (B) framing
> fill bias under look-at jitter, (C) fixture/test hardening, (D) frontend defaults drift,
> (E) documentation debt. Same hard rules as always (`.ai/AGENTS.md`): bpy only in
> `scene.py`, new logic in pure modules, every step lands with tests in the bpy-free lane
> unless stated otherwise. Each step is independently committable.

---

## Track A — `--workers N` actually runs in parallel (bug)

**Current behavior:** `render()` in `render.py` launches workers with `subprocess.run(command,
check=True)` inside a `for` loop. `subprocess.run` blocks until the child exits, so workers
execute one after another. `--workers 4` is strictly slower than `--workers 1` (4× bpy
startup + 4× OBJ import, fully serialized).

### A.1 Replace `run`-in-loop with `Popen` + wait

In `render.py`, coordinator branch (`workers > 1`):

```python
processes: list[tuple[int, subprocess.Popen[bytes]]] = []
for index in range(workers):
    command = _worker_command(...)
    processes.append((index, subprocess.Popen(command)))

failed: list[int] = []
for index, process in processes:
    if process.wait() != 0:
        failed.append(index)
if failed:
    raise WorkerRenderError(failed)  # see A.3
```

Notes:
- Launch **all** `Popen` calls before the first `wait()`. No output capture needed — workers
  inherit stdout/stderr, which keeps the existing per-frame progress lines visible (now
  interleaved; acceptable, and `run.json` is the authoritative record anyway).
- Do not use `concurrent.futures` — there is nothing to gain over plain `Popen` here and it
  obscures the "separate OS processes because bpy is a process-global" rationale.

### A.2 Fail fast: terminate siblings on first failure

A fully-failed worker should not leave the others rendering for an hour. After launching,
poll instead of blocking serially:

```python
import time

remaining = dict(processes)          # index -> Popen
failed: list[int] = []
while remaining:
    for index, process in list(remaining.items()):
        code = process.poll()
        if code is None:
            continue
        del remaining[index]
        if code != 0:
            failed.append(index)
    if failed and remaining:
        for process in remaining.values():
            process.terminate()      # SIGTERM; bpy exits cleanly enough for our purposes
        for process in remaining.values():
            process.wait()
        remaining.clear()
    if remaining:
        time.sleep(0.2)
```

Keep this loop small and inline (or as a private `_wait_for_workers(processes) -> list[int]`
helper so it is unit-testable without real subprocesses).

### A.3 `WorkerRenderError` in `errors.py`

```python
class WorkerRenderError(RuntimeError):
    """Raised when one or more parallel render workers exit non-zero."""

    def __init__(self, failed_worker_indices: list[int]) -> None:
        self.failed_worker_indices = failed_worker_indices
        indices = ", ".join(str(i) for i in failed_worker_indices)
        super().__init__(f"render worker(s) failed: {indices}")
```

Coordinator must **not** run `merge_run_metadata` or the dataset writer when workers failed —
a partial dataset that looks complete is worse than a crash. Partial
`run.frames.worker_*.json` files are left in place for debugging.

### A.4 Cap workers at frame count + remove dead code

- `workers = min(workers, cfg.camera.n)` before spawning — workers beyond `n` get empty
  index lists today (harmless but wasteful: each still pays bpy + OBJ import startup).
- Delete the unreachable `if workers < 1: raise` inside the `workers > 1` branch (the typer
  option already enforces `min=1`).

### A.5 Tests (bpy-free, monkeypatched `Popen`)

The existing coordinator test only asserts command construction, which is why A's bug went
unnoticed. Add to `test_parallel_render.py`:

- `test_coordinator_starts_all_workers_before_waiting`: monkeypatch `subprocess.Popen` with a
  fake that appends `("start", index)` to a shared event list on construction and
  `("wait", index)` on `poll()`/`wait()`. Assert every `start` event precedes the first
  `wait`-driven completion handling — i.e. `events.index(("start", N-1)) <
  events.index(first_completion)`.
- `test_coordinator_raises_and_terminates_on_worker_failure`: first fake worker returns
  exit code 1; assert `WorkerRenderError` lists index 0, `terminate()` was called on the
  others, and neither `run.json` nor `dataset/` was created.
- `test_coordinator_caps_workers_at_frame_count`: `camera.n = 2`, `workers=8` → exactly 2
  `Popen` constructions.

---

## Track B — framing: jitter must not change apparent object size (fidelity)

**Current behavior:** `render.py` calls `scene.move_camera(location, look_at=framing.look_at,
fit_margin=...)`. `Scene._fit_camera_to_target` computes the bounding radius **about the
jittered look_at**, which is larger than about the object center, so the camera is pushed
back farther than the sampled `fill` implies. The same happens again in
`_refit_camera_for_current_render_settings` at render time (it refits from the stored,
jittered `_camera_look_at`). Result: realized fill is biased low, increasingly so at high
`center_jitter`; the realized pixel offset also deviates from `framing.py`'s intent. Labels
stay correct (mask-derived) — only the configured distribution is distorted.

**Fix principle:** *fit about the object anchor, aim at the jittered point.* Distance (size)
is determined relative to the pre-jitter look_at; orientation alone produces the in-frame
translation. This also makes `framing.fitted_camera_distance` (which already uses the
pre-jitter radius) consistent with what the scene actually does.

### B.1 `Scene.move_camera` gains `fit_about`

```python
def move_camera(
    self,
    location: tuple[float, float, float],
    look_at: tuple[float, float, float] = (0.0, 0.0, 0.0),
    fit_target: bool = True,
    fit_margin: float = 1.2,
    fit_about: tuple[float, float, float] | None = None,
) -> bpy.types.Object:
```

- `fit_about` defaults to `look_at` (today's behavior — every existing caller and test is
  unaffected).
- `_fit_camera_to_target` changes: `radius` is computed about `fit_about`; the pushback ray
  is `fit_about - requested_location` (push the camera back along the line to the **anchor**,
  not to the aim point); `distance = max(|fit_about - location|, min_distance)`;
  `camera.location = fit_about - direction.normalized() * distance`. Aiming at `look_at`
  stays in `_point_camera_at` afterwards, unchanged.
- Store `self._camera_fit_about` alongside the existing `_camera_look_at` /
  `_camera_fit_margin`, and use it in `_refit_camera_for_current_render_settings` — this is
  the second call site of the bias and must not be missed.
- Mirror the parameter on `add_camera` for symmetry (pass-through).

### B.2 `render.py` passes the anchor

```python
scene.move_camera(
    location=pose.location,
    look_at=framing.look_at,        # jittered: orientation only
    fit_margin=framing.fit_margin,
    fit_about=pose.look_at,         # pre-jitter: size only
)
```

No change to `framing.py` — its `fitted_camera_distance` already computes about the
pre-jitter look_at, which is now exactly what the scene does, so the jitter pixel-scale math
(`d·tan(fov/2)·center_jitter`) becomes accurate rather than approximate.

### B.3 Tests

- **Pure (`test_scene_camera_fit.py`-adjacent, math-level):** factor the pushback geometry
  (`anchor`, `location`, `radius`, `fov`, `margin` → fitted location) into a small pure
  helper if it isn't already extractable; assert the fitted location is independent of the
  aim point.
- **bpy (`test_render_orientation.py`-adjacent, the sharp one):** render the committed cube
  fixture twice at the same pose, `fill = 0.4`, once with `center_jitter = 0` and once with
  `jitter_uv = (0.9, 0.0)` forced (call `jitter_look_at` directly to make the offset
  deterministic instead of seeding). Assert the mask **height** differs by at most 2 px
  between the two renders (size invariance under jitter), and the mask **center x** moved by
  at least 20% of the frame (jitter actually translates). This test fails against the
  current code and passes after B.1/B.2 — write it first.
- **Distribution sanity (cheap, mask-free):** in `test_framing.py`, assert
  `fitted_camera_distance` for `fill=0.5` equals the scene helper's distance for any jitter
  magnitude (ties the two modules together explicitly).

---

## Track C — fixtures: stop depending on possibly-uncommitted `.obj` files

**Current behavior:** `test_label_parity.py` and `test_orientation_parity.py` reference
`tests/fixtures/two_offset_cubes.obj` and skip when absent. `--require-bpy` hard-fails only
on missing **bpy**, not missing fixtures — so if the file was never committed, the
multi-mesh parity guards pass-by-skipping in CI, the exact failure mode the bpy lane exists
to prevent.

### C.1 Verify what is actually committed (one command, do this first)

```bash
git ls-files tests/fixtures
```

Expected: `asymmetric_y_up.obj`, `asymmetric_z_up.obj`, `two_offset_cubes.obj`,
`textured_cube/cube.mtl` (+ its `.obj`/texture). Anything missing → C.2 makes the two-cube
case moot; for the asymmetric fixtures, commit them (they are a few hundred bytes of text).

### C.2 Generate the two-cube fixture in-test instead of committing it

The fixture is trivial deterministic text — generating it removes the "is it committed?"
question permanently. Add `tests/fixture_factories.py`:

```python
def write_two_offset_cubes_obj(path: Path) -> Path:
    """Two unit cubes: one centered at origin, one offset +3 on X (Z-up)."""
    lines: list[str] = []
    offsets = ((0.0, 0.0, 0.0), (3.0, 0.0, 0.0))
    for cube_index, (ox, oy, oz) in enumerate(offsets):
        lines.append(f"o cube_{cube_index}")
        for sx in (-0.5, 0.5):
            for sy in (-0.5, 0.5):
                for sz in (-0.5, 0.5):
                    lines.append(f"v {ox + sx} {oy + sy} {oz + sz}")
        base = cube_index * 8 + 1
        for a, b, c, d in _CUBE_QUADS:           # the 6 quads of a unit cube
            lines.append(f"f {base+a} {base+b} {base+c} {base+d}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
```

Important detail: emit **two `o` groups** so Blender's importer creates two separate mesh
objects — that is the property the multi-mesh tests exist to exercise. Add a quick bpy
assertion in the test: `assert len(scene.targets) == 2` (guards against a future importer
default that merges objects, which would silently neuter the test).

Update both tests to call the factory with `tmp_path` and drop the `pytest.skip` /
`TWO_CUBE_FIXTURE_OBJ` constant. The committed asymmetric fixtures stay as-is (they guard
orientation, where a hand-authored asymmetric shape is the point); only intentionally
optional big assets under `test-obj/` keep their skips.

### C.3 (Small, optional) make required-fixture skips loud in the bpy lane

If C.1 reveals missing asymmetric fixtures and you choose to keep any committed-fixture
dependencies: in `conftest.pytest_runtest_setup`, when `--require-bpy` is set and a test
calls `pytest.skip` with a message starting `"fixture not found"`, convert to `pytest.fail`.
Implement by replacing inline skips with a tiny helper
`require_fixture(path, *, config)` in `tests/orientation_checks.py` that fails instead of
skipping when the option is set. Skip this step if C.2 removes the last required-fixture
skip — don't build machinery without a client.

---

## Track D — frontend defaults: single source of truth (drift risk)

**Current behavior:** `defaultConfig.ts` + `serializeConfig.ts:mergeConfigDefaults` duplicate
every pydantic default in TypeScript. Next schema change that lands only in `config.py` makes
the SPA's YAML preview silently lie — the exact "silently saved defaults" failure the preview
was built to prevent. Hard rule 2 already names the remedy: *if the frontend needs a computed
value, add an endpoint.*

### D.1 `GET /api/config/defaults`

In `web/api.py`:

```python
@router.get("/config/defaults", response_model=RembrandtConfig)
def config_defaults() -> RembrandtConfig:
    """Schema defaults for the SPA (single source of truth: pydantic)."""
    return RembrandtConfig(
        object=ObjectConfig(path=""),
        camera=CameraConfig(n=10),
    )
```

The two required fields get the same placeholder values `defaultConfig.ts` uses today, so the
payload is exactly the current TS object. Note `CameraConfig(n=10)` runs the existing
validators — fine. Add a `test_web_api.py` case asserting the response parses back through
`RembrandtConfig.model_validate` and that `camera.seed`, `framing.fill_range`, etc. match the
pydantic defaults (trivially true, but pins the route's shape).

### D.2 Frontend consumes the endpoint

- `api.ts`: add `fetchConfigDefaults(): Promise<RembrandtConfig>`.
- App startup: fetch once, hold in state, pass down. `createDefaultConfig(objectPath)`
  becomes a thin overlay: `{ ...fetchedDefaults, object: { ...fetchedDefaults.object, path } }`.
- `mergeConfigDefaults` takes the fetched defaults as a parameter instead of importing
  `createDefaultConfig`. Delete the hand-written default literals from `defaultConfig.ts`
  entirely — keep only the overlay helper. The serializer (`configToYaml`) is untouched;
  it remains display-only.
- Loading state: until defaults arrive, disable the Save bar (the preview cannot be honest
  without them). One `isLoading` boolean; no spinner ceremony needed.

This is the only track touching the frontend; follow `ct-frontend-design` conventions as
AGENTS.md already mandates.

---

## Track E — documentation debt (high value for agent workflows)

### E.1 `.ai/AGENTS.md` — the priority item

This file is read by every Claude Code session and currently instructs agents that the
shipped features do not exist. Update:

- **"What this project is"**: replace "Today the render step writes PNG frames only. YOLO
  labels … are planned follow-up work — do not assume they exist" with the actual pipeline:
  render → alpha-mask YOLO labels → train/val layout + `data.yaml` → training handoff files;
  `--frames-only` opts out; `--workers N` parallelizes; `--stats` prints label distributions.
- **Module map**: add `annotations.py`, `dataset.py`, `framing.py`, `postfx.py` with their
  one-line roles and `NO bpy` markers; delete the stale "planned-but-not-yet-built modules
  … do not exist yet" paragraph.
- **Hard rules**: add two new load-bearing invariants:
  7. *Labels come from the rendered alpha mask, never from projected geometry at runtime*
     (vertex projection exists only as a bpy-lane parity test).
  8. *Post-fx must be geometry-preserving and applied after label extraction* — anything
     that moves pixels belongs in 3D, not in `postfx.py`.
- **Gotchas**: EEVEE-needs-GPU-headless (with the `RenderEngineUnavailableError` name);
  "when framing is enabled, `distance_range` is a lower bound — sampled `fill` controls
  apparent size"; "labels default ON: `background.mode: none` now composites over
  `background.color` instead of the Blender world".
- **Seeds**: extend the seed-independence sentence to the full set: camera, background,
  light, framing, postfx, split.

### E.2 `README.md` — remove the contradictions, document the headline feature

- Delete the intro sentence "The current render command writes frames only; YOLO labels …
  planned follow-up work."
- Replace the Quick Start "Render" section's output description: default output is
  `<output.dir>/<stamp>/dataset/` with `images/{train,val}`, `labels/{train,val}`,
  `data.yaml`, `train_yolo.py`, `README.md`; flat `frame_*.png` only with `--frames-only`.
  Show the two-command happy path:
  ```bash
  rembrandt-render ./configs/dataset.yaml
  cd output/<stamp>/dataset && pip install ultralytics && python train_yolo.py
  ```
- Delete the stale "`train_val_split` is reserved for the future dataset writer" line; it is
  consumed now. Document `output.split_seed` next to it.
- Add short config-reference subsections for `labels:`, `framing:`, `postfx:` mirroring the
  existing `background:`/`light_randomization:` style (defaults + one sentence each,
  including the `min_visible_pixels` → empty-label-negative behavior).
- One sentence under backgrounds: cast shadows are not preserved in composited mode (no
  shadow catcher yet) — sets expectations for the known floating-object look.

### E.3 `configs/dataset.yaml` — regenerate the de-facto schema example

Regenerate via `dump_config(RembrandtConfig(...))` from a config that exercises the new
sections, then hand-add the commented optional blocks in the existing style:
`object.class_name`/`class_id` set, `labels:` explicit, `framing:` explicit, `postfx:` and
`light_randomization:` present-but-commented. This file is what people copy; it should show
the schema that exists.

### E.4 Consistency nit while in `render.py` docs

`README` says output-dir resolution follows "the same resolution order as object paths" —
not exactly true: `resolve_object_path` falls back to CWD-relative, `resolve_output_dir`
falls back to config-relative (sensible for a directory that does not exist yet). Keep the
behavior, fix the sentence: "absolute → as-is; else config-relative; else CWD if it exists
there; for a new directory, created next to the config."

---

## Sequencing & sizing

| Order | Track | What lands | Size |
|---|---|---|---|
| 1 | A | True parallelism + fail-fast + tests that would have caught it | S |
| 2 | B | `fit_about` anchor; jitter no longer shrinks objects; invariance test | S–M |
| 3 | C | `git ls-files` audit; generated two-cube fixture; loud-skip helper if needed | S |
| 4 | E | AGENTS.md first, then README, then sample config | S |
| 5 | D | `/api/config/defaults` + frontend consumes it; TS default literals deleted | M |

A and B are independent and can be parallel branches. E.1 (AGENTS.md) can ship the same day
regardless of the rest — it is currently the most misleading artifact in the repo for your
own workflow. D is last because it is the only multi-file frontend change and nothing else
depends on it.

Definition of done (all tracks): `pytest -m "not bpy" -q`, `pytest -m bpy --require-bpy -q`,
`ruff check`, `ruff format --check`, `mypy --no-sqlite-cache src`, `cd frontend && yarn
typecheck && yarn build` — all clean, per AGENTS.md.