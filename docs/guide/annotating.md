# Annotating training data

To train a segmentation model you need training data, and the first step in Ais is to annotate a little bit of your own data. We do this very sparsely: only part of a slice, and only in one or a couple of tomograms. Typically you then train a network — you can watch it learn by training in the GUI, or train on a cluster — assess what works well and what does not, refine the training data a bit where needed, and then continue training with an updated training set. After a little while you'll find that the network segments your feature of interest well, perhaps as well as you would do by hand.

With experience, you can generate a useful model in less than 30 minutes. This page lists some tips and tricks for effective annotation.

<video controls muted loop autoplay playsinline width="100%">
  <source src="../../res/annotation.mp4" type="video/mp4">
</video>
<p style="text-align: center; font-style: italic; color: var(--md-default-fg-color--light); margin-top: 0.5em;">Annotating several features in a single tomogram.</p>

## What data does the model see during training?

When annotating in Ais you do two things: you place **boxes** and you **draw** annotations on the tomogram. Only the image regions _within the boxes you place_ are used during training.

The single most important thing: wherever you place a box, you must _unambiguously_ annotate your feature of interest. Suppose you're segmenting ribosomes, and a box in some region contains five of them — you must annotate all five, in a consistent manner. If you annotate four but miss the fifth, you are telling the network that those first four particles are ribosomes and the fifth is not. That is ambiguous, and it hurts learning.

A second important thing: the shape in which you annotate is the shape the network learns to produce. You can place a simple dot on each ribosome, or outline it more precisely — choose whatever works best for your goal. For microtubules and filament tracing, for example, it can be useful to annotate not the full ~25 nm width of the filament but only the lumen. That way, microtubules in tightly packed bundles are still segmented as individual filaments (and you can always postprocess the label to cover the full width). For ribosomes, I like to roughly annotate their whole shape. That way the resulting visualisations look a bit more realistic. But you can do whatever you like.

??? note "Positive and negative boxes"
    In earlier versions of Ais — and in our older tutorials — we drew a more explicit distinction between *positive* and *negative* boxes:

    - A **positive box** contains density **and** an annotation: the network learns your feature as foreground and the rest of the box as background.
    - A **negative box** contains density but **no** annotation: everything inside it is assumed to be background — a pure counterexample.

    When you export training data you can choose which annotated features to use as positives and which as negatives.

    In practice, though, **you can ignore this distinction if you want.** If you are annotating ribosomes and you place a box in a region that happens to contain none — and you correctly leave it un-annotated — that box is a perfectly valid training example either way. An empty box teaches the network what background looks like regardless of whether it is formally treated as "positive" or "negative".

    Where the distinction *is* convenient is for shared counterexamples. Suppose you want models for membranes, ribosomes, and microtubules. You can annotate a fourth thing called something like `Junk`, and use it only to place boxes over bad image regions — contamination, void, carbon film. Those boxes then serve as counterexamples to all three real models at once, so you don't have to place empty boxes separately for each.

## How much do I need to annotate?

Our advice is: **be lazy**. Start with just one or two tomograms and annotate for two to five minutes. Then train a small network — VGGNet M is a good option and will train in just a few minutes — and inspect the results. Wherever the network makes a mistake is a region where it would help to add an annotation. Toggle back and forth between the Models and Annotation tabs (shortcut: use the `1` and `2` keys to toggle between tabs) to improve the training data, do this for a few more minutes, and test again.

Achieving good output in machine learning is very iterative. The alternative — doing hours of annotation and then training once, in the hope of getting a perfect model — never works.

??? question "What box size should I use?"
    The _receptive field_ of a network is the region of the input image that a single output pixel depends on — in other words, how much surrounding context the network gets to look at when it decides whether a pixel belongs to your feature. Its size is determined primarily by the network architecture: how deep the network is, its kernel sizes, and how much it downsamples.

    The second most important thing is the box size you use during training. A network with a large receptive field, trained on a small box, will not be able to use all of the context it is built to see. In general, *a big box is better*, but the annotation cost per box does of course go up. For fine grained features like ribosomes, membranes, microtubules, nucleosomes, etc., start with a box size of 100 to 150 nm; if you're working at 10 A/px, 128 pixel boxes are good. For larger things like entire mitochondria or vesicles, use a box size of around 500 nm. At 10 A/px, that'd be a 512 pixel box, which is a bit much for training (training is slower for larger boxes). That's why you can bin the training data when you export it. For mitochondria, we would annotate a 512 pixel box in our 10 A/px tomograms and then during export (`ais extract -f mitochondrion -size 512 -bin 5`) bin the images down to 50 A/px. Training and inference are then done at 50 A/px, which will end up being very fast.

??? tip "Difficult feature? A small annotated box is enough"
    Some features are simply awkward to annotate — an intricate or convoluted shape that would take a long time to trace across a full-size box. You don't have to. Place a *smaller* box — 32 or 64 pixels, or whatever area you are comfortable annotating completely — around just a patch of the feature, even when the rest of your boxes are the usual larger size.

    On export, Ais pads that box out to the training box size with **ignore** labels around the edge: pixels that are neither foreground nor background and are simply skipped when the training loss is computed. The network still learns from the small region you _did_ annotate, while your other, full-size boxes contribute as normal. A lot of ignored border is completely fine — it costs you nothing, and it saves you from having to perfectly annotate a large, complicated area.

??? question "Should I place many overlapping boxes?"
    No. Overlapping boxes do not provide much new information for training. Convolutional neural networks are, in principle, translation invariant, so translational jitter in the boxes is not very important. A little overlap does not hurt performance — but there is no benefit to piling boxes on top of one another.

??? question "Do I have to re-define feature names and colours every time I add an annotation?"
    No — check out the [feature library](../features/feature_library.md).

## How do I annotate faster?

### The active-contouring brush

Turn on *flood* mode to use the active-contouring brush. Instead of tracing a feature by hand, the brush snaps to the edges of the structure under your cursor — adjust its sensitivity and it will follow membranes, filaments, and other well-defined boundaries automatically.

<video controls muted loop autoplay playsinline width="100%">
  <source src="../../res/active_contouring.mp4" type="video/mp4">
</video>
<p style="text-align: center; font-style: italic; color: var(--md-default-fg-color--light); margin-top: 0.5em;">The active-contouring brush snapping to feature edges in flood mode.</p>

!!! tip
    The image **filters** (Gaussian, Sobel, invert, contrast, …) are for annotation only — the network just sees the original tomogram — but a filtered image can give the active-contouring brush clearer edges to snap to. See [Filters & display](../features/filters.md).

See [The active-contouring brush](../features/active_contouring.md) for more.

### Model-assisted annotation

Once you've trained a quick first model, model-assisted annotation becomes very useful. Screen the model's output on a fresh region, copy the parts it got right straight into your training annotations, and fix only the mistakes by hand. This turns annotating into editing — much faster than drawing everything from scratch. In the model tab, right-click the model you want to copy the output of, and in 'copy to annotation', select the destination annotation.

<video controls muted loop autoplay playsinline width="100%">
  <source src="../../res/model_assisted_annotation.mp4" type="video/mp4">
</video>
<p style="text-align: center; font-style: italic; color: var(--md-default-fg-color--light); margin-top: 0.5em;">Copying a model's output into the training annotations, then correcting the mistakes.</p>

See [Model-assisted annotation](../features/model_assisted_annotation.md) for more.

??? question "Can I save annotated tomograms?"
    Yes. Annotations are saved as `.scns` files. These are pickle files that hold all of your annotations and their metadata — but **not** the tomogram density itself. Instead, an `.scns` file stores a *link* to the original `.mrc` file.

    This means that if you move the original `.mrc` after saving, the link breaks, and you'll need to re-link it the next time you open the `.scns`. You can do this from the top menu bar under **File manager → Open file manager**, or by right-clicking the dataset in the Datasets panel and choosing **Relink dataset**.

## Controls

| Action | Control |
| --- | --- |
| Draw | left mouse button |
| Erase | right mouse button |
| Place box | `SHIFT` + left mouse button |
| Remove box | `SHIFT` + right mouse button |
| Change brush size | hold `CTRL` + scroll |
| Toggle active-contouring brush | `F` |
| Move view | middle mouse button + drag |
| Zoom view | scroll + hold `SHIFT` |
| Recenter view | `space` |
| Browse slices | scroll, or `←` / `→` |
| Toggle contrast | `I` |
| Toggle interpolation | `SHIFT` + `I` |
| Toggle filters | `Z` |
| Set active feature | click it in the Features panel, or `W` / `S` |
| Set active tomogram | `↑` / `↓` |
| Switch tab (Segmentation / Models / Export / Render) | `1` / `2` / `3` / `4` |
| Quick-save annotated tomogram | `CTRL` + `S` |
