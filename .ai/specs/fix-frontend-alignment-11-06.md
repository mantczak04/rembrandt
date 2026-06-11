Fix a layout problem in the Rembrandt SPA frontend (React 19 + TypeScript + CSS modules,
Vite, in `frontend/`). Read `.ai/AGENTS.md` first. This is a layout-only change: no new
features, no API changes, no changes to config state handling or `serializeConfig.ts`.

## Problem

`frontend/src/App.tsx` renders three stacked rows inside `.layout` (which is
`height: 100vh; overflow: hidden` in `frontend/src/App.module.css`):

1. `header`
2. `.main` — a 2-column grid: `.viewportPane` (Three.js canvas) + `.controlsPane`
   (scrollable form sections)
3. `.savePane` — full-width `<SaveBar>` with filename input, Save button, a
   "Show YAML preview" checkbox (default ON), and an unbounded `<pre class={yamlPreview}>`

Because row 3 is full-width and the YAML preview has no height cap and defaults to open,
the save section consumes ~20% or more of the vertical space, shrinking the viewport and
making the page feel scrollable. The 3D viewport should get all remaining height.

## Required end state

1. Delete the third row. Remove the `.savePane` wrapper from `App.tsx` and the `.savePane`
   rule from `App.module.css`. `.main` becomes the last child of `.layout` and keeps
   `flex: 1; min-height: 0`.

2. Move `<SaveBar>` into the right column as a pinned footer. Restructure the right column
   in `App.tsx` to:

```tsx
   <div className={styles.controlsColumn}>
     <div className={styles.controlsScroll}>
       <Controls ... />   {/* unchanged props */}
     </div>
     <div className={styles.saveFooter}>
       <SaveBar config={config} disabled={mesh === null} />
     </div>
   </div>
```

   In `App.module.css`, replace `.controlsPane` with:

```css
   .controlsColumn {
     min-height: 0;
     display: flex;
     flex-direction: column;
   }
   .controlsScroll {
     flex: 1;
     min-height: 0;
     overflow: auto;
   }
   .saveFooter {
     flex-shrink: 0;
     border-top: 1px solid <existing panel border color from Controls.module.css>;
     padding-top: 0.75rem;
   }
```

   Keep the existing grid columns on `.main` (`minmax(0, 1.6fr) minmax(18rem, 1fr)`).

3. Make `SaveBar` compact (edit `frontend/src/controls/SaveBar.tsx` and
   `frontend/src/controls/Controls.module.css`):
   - Filename input and Save button on ONE row: a `.saveRow` grid with
     `grid-template-columns: minmax(0, 1fr) auto; gap: 0.5rem; align-items: center;`.
     Keep the existing input/button classes for visual consistency; drop the standalone
     "Filename" label in favor of `aria-label="Config filename"` on the input (keep the
     `placeholder="dataset.yaml"`).
   - Keep the `h2.sectionTitle` ("Save config") but it may share a row with the
     "Show YAML preview" checkbox to save a line (title left, checkbox right,
     `display: flex; justify-content: space-between; align-items: center;`).
   - `savedPath` / `error` messages stay below the row, unchanged.

4. Tame the YAML preview so it can never change the page height:
   - Change the `showPreview` useState default from `true` to `false`.
   - Add to `.yamlPreview`: `max-height: 32vh; overflow: auto; margin: 0.5rem 0 0;`
     (keep existing font/background styles). The preview must scroll internally;
     opening it must not move the viewport pane or introduce page scroll.

5. No page-level scrolling at any reasonable desktop size (≥ 1280×720): `.layout` stays
   `100vh; overflow: hidden`; the only scroll containers are `.controlsScroll` and
   `.yamlPreview`. The viewport canvas must visibly grow taller compared to before
   (it now gets the full height of `.main`).

6. Mobile breakpoint: inside the existing `@media (max-width: 900px)` block, allow the
   single-column layout to scroll the page naturally — set `.layout { height: auto;
   overflow: visible; }` there and give `.viewportPane` a fixed `height: 45vh` so the
   canvas stays usable.

## Constraints

- Do not modify `Controls.tsx`, `Viewport.tsx`, `serializeConfig.ts`, or any backend file.
- Do not introduce new dependencies or inline styles; CSS modules only.
- `Viewport` already resizes via its own observer — do not add resize handling in App.

## Acceptance checks

- `cd frontend && yarn typecheck && yarn build` pass.
- Manual check via `make dev`: (a) no page scrollbar with preview closed AND open;
  (b) Save button visible without scrolling the controls column; (c) opening the YAML
  preview scrolls internally; (d) controls column scrolls independently; (e) at
  < 900px width the page scrolls vertically and nothing is clipped.