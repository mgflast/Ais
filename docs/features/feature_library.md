# The feature library

Every feature you annotate has a name, a colour, a box size, a brush size, and a set of rendering settings. The feature library stores these as reusable presets, so you define a feature once and load it whenever you annotate it again.

<video controls muted loop autoplay playsinline width="100%">
  <source src="../../res/feature_library.mp4" type="video/mp4">
</video>
<p style="text-align: center; font-style: italic; color: var(--md-default-fg-color--light); margin-top: 0.5em;">Loading a saved feature's settings when adding a new annotation.</p>

## What it stores

Each library entry keeps, per feature:

- **Annotation settings** — name, colour, box size, brush size, and overlay opacity.
- **Rendering settings** — the isosurface level, dust size, and transparency used in the Render tab.

Libraries are saved to `~/.Ais/feature_library.txt` and persist across datasets and sessions. You can keep several named libraries and switch which one is active.

!!! note "Multiple libraries"
    Keep a large catalogue for a big project, and a small, focused library — just the handful of features a given analysis needs — for that analysis. You switch the active library from the same menu.

## How to use it

1. Open the library from **Settings → Feature library → Open library**. Add features and set their annotation and rendering settings — the panel toggles between the two.
2. When you add an annotation, right-click its title to pick a saved feature; its settings load in one click.

!!! note "The icon buttons"
    The three round icon buttons at the top-left of the window, just right of the menu bar, are shortcuts: a sparkle toggles party mode, a moon toggles dark mode, and the third — the library icon — opens the feature library.

!!! tip
    For a project with many features, set them all up in the library first. Every new annotation then starts from consistent names, colours, and box sizes across the whole dataset.
