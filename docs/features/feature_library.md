# The feature library

Every feature you annotate has a name, a colour, a box size, a brush size, and a set of rendering settings. The feature library stores these as reusable presets, so you define a feature once and load it whenever you annotate it again.

<video controls muted loop autoplay playsinline width="100%">
  <source src="../../res/feature_library.mp4" type="video/mp4">
</video>
<p style="text-align: center; font-style: italic; color: var(--md-default-fg-color--light); margin-top: 0.5em;">Loading a saved feature's settings when adding a new annotation.</p>

Open the library from **Settings → Feature library → Open library**. Add features and set their annotation and rendering settings — the panel toggles between the two:

- **Annotation settings** — name, colour, box size, brush size, and overlay opacity.
- **Rendering settings** — the isosurface level, dust size, and transparency used in the Render tab.

Libraries are saved to `~/.Ais/feature_library.txt` and persist across datasets and sessions. You can keep several named libraries and switch which one is active.

When you add a new annotation to a tomogram, the drop-down menu for that feature's title lists your presets. Select one to set its colour, brush size, and box size to the predefined values.

When you load a model whose name is in the feature library, it takes that feature's colour. In the Render tab, a feature whose name is in the library is likewise rendered with that feature's colour, dust size, and isosurface level.

!!! note "The icon buttons"
    The three round icon buttons at the top-left of the window, just right of the menu bar, are shortcuts: a sparkle toggles party mode, a moon toggles dark mode, and the third — the library icon — opens the feature library.

!!! tip
    For a project with many features, set them all up in the library first. Every new annotation then starts from consistent names, colours, and box sizes across the whole dataset.
