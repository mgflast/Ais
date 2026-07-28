# Model-assisted annotation

Drawing every annotation by hand is slow. Once you've trained even a rough first model, you can let it do most of the work: test its output on a slice, copy that whole slice into the Annotation tab, and fix only the mistakes by hand. This turns annotating from *drawing* into *editing* — much faster. It also means you annotate and test at the same time, and can add training data exactly where the model is still inaccurate.

<video controls muted loop autoplay playsinline width="100%">
  <source src="../../res/model_assisted_annotation.mp4" type="video/mp4">
</video>
<p style="text-align: center; font-style: italic; color: var(--md-default-fg-color--light); margin-top: 0.5em;">Copying a model's output into the training annotations, then correcting the mistakes.</p>

## How to use it

1. Train a quick first model — see [Training a model](../guide/training.md). It doesn't have to be good; it only has to be a starting point.
2. Apply it to a tomogram and play with the threshold — and, optionally, the test-time augmentation multiplicity — until the output looks about right. See [Assessing model performance](../guide/assessing_performance.md).
3. **Right-click the model's panel** and open *copy output to annotation*, then pick the destination annotation from the list of features.
4. Ais copies the output into that annotation and switches you to the **Annotation** tab. Now correct it: erase the false positives, and draw in whatever the model missed.
5. Add and correct a few boxes, then train again. A useful rule of thumb: annotate about 20 boxes and train a first version; then use that model to assist you up to roughly 50 boxes, export, and train again — and see how much better it gets.

!!! note "What gets copied"
    *Copy output to annotation* transfers only the slice you're looking at, and only the pixels above the current threshold. Take care when copying onto a slice you have already annotated: it overwrites your hand-drawn labels there, though any boxes you placed stay in place.
