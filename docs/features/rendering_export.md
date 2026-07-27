# Rendering & visualisation

The **Render** tab has a built-in 3D renderer for inspecting segmentation results.

<video controls muted loop autoplay playsinline width="100%">
  <source src="../../res/render_3d.mp4" type="video/mp4">
</video>
<p style="text-align: center; font-style: italic; color: var(--md-default-fg-color--light); margin-top: 0.5em;">In the Render tab in Ais, segmentations belonging to a tomogram are automatically detected and displayed in 3D. In this example we inspect microtubule (yellow) and microtubule inner protein (blue) segmentations.</p>

!!! tip
    - Turn off *Settings → Rendering → Wait to render* to have segmentations render in 3D automatically.
    - If Ais fails to automatically detect the segmentation volumes, add a *search directory* (the folder containing the segmentation `.mrc` files) under *Settings → Rendering → Search directories*.
    - In the feature library (*Settings → Feature library → Open feature library*) you can preset the isosurface level, dust size, and transparency for each feature.

## Export to ChimeraX or Blender

The visualisation options in Ais are less extensive than those in ChimeraX. For quick inspection Ais is useful, but if you want to visualise segmentations and particles in more detail we recommend ChimeraX + ArtiaX. Using the **Export 3D scene** tab you can forward 3D models to ChimeraX or Blender, provided you have set up the paths to these applications under *Settings → 3rd party applications*.
