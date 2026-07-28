# The active-contouring brush

Tracing a feature by hand is slow. The active-contouring brush — the **flood** toggle in a feature's panel, or the `F` key — fills a whole region in one stroke instead. It reads the brightness of the pixel under your cursor and grows outward into the connected area of similar-brightness pixels, stopping where the brightness changes. That makes it snap to the edges of membranes, filaments, granules, and other well-defined boundaries.

<video controls muted loop autoplay playsinline width="100%">
  <source src="../../res/active_contouring.mp4" type="video/mp4">
</video>
<p style="text-align: center; font-style: italic; color: var(--md-default-fg-color--light); margin-top: 0.5em;">The active-contouring brush filling a feature in one stroke, snapping to its edges.</p>
<!-- record a fresh clip: flood toggled on, one-stroke fill on a membrane, then a sensitivity adjustment -->

## How to use it

1. Select the feature you want to annotate, then turn on **flood** (the checkbox in its panel, or press `F`).
2. Set the **brush size** with `CTRL` + scroll — this caps how far a single stroke can reach (shown in px and nm).
3. Draw over the feature with the left mouse button; each stroke fills the connected region of similar brightness under the cursor. Erase as normal with the right mouse button.

!!! note "What the sensitivity does"
    A **sensitivity** slider appears when flood is on (the `-` / `=` keys adjust it too). The brush fills pixels whose brightness falls within a band around the one under your cursor: lower sensitivity widens that band, filling more but bleeding past weak edges; higher sensitivity narrows it, filling less and stopping at fainter boundaries.

!!! tip
    Flood works best on a clean image. Blur the tomogram with the [filters](filters.md) so a whole membrane or granule reads as one contiguous region and fills in a single click. Filters change only what you and the brush see — the network still trains on the raw tomogram.
