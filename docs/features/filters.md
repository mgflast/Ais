# Filters & display

In the **Filters** section of the GUI sidebar you can set up and combine different filters to apply to a tomogram. These are for visualisation during annotation only — they do not affect the data that goes into training. When you export a training dataset, Ais uses the unfiltered pixel data from the tomogram itself.

<video controls muted loop autoplay playsinline width="100%">
  <source src="../../res/filters.mp4" type="video/mp4">
</video>
<p style="text-align: center; font-style: italic; color: var(--md-default-fg-color--light); margin-top: 0.5em;">Filters are applied on the fly for visualisation; the data used for training is always the raw, unfiltered tomogram.</p>

When using the active-contouring brush, it can help to blur the tomogram quite a bit. This removes noise and makes contiguous areas — a membrane, a granule, a ribosome — easier for the brush to snap to in one click.

## Shortcuts

| Action | Shortcut |
| --- | --- |
| Toggle all filters on/off | `Z` |
| Invert tomogram contrast | `I` |
| Toggle nearest-neighbour / linear interpolation | `SHIFT` + `I` |
| Toggle autocontrast | `A` |
