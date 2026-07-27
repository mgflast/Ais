# Training a model

Once you've [annotated](annotating.md) a bit of data, the next step is to turn those annotations into a training dataset and train a network on it. You can watch a model train live in the GUI, or do the training on a multi-GPU node with [`ais train`](../cli/train.md). You don't need a cluster to get started: consumer GPUs are fine — we often run Ais on a laptop (a Quadro P1000) — and you can even train and test models on the CPU alone, if a little slowly.

<video controls muted loop autoplay playsinline width="100%">
  <source src="../../res/training_run.mp4" type="video/mp4">
</video>
<p style="text-align: center; font-style: italic; color: var(--md-default-fg-color--light); margin-top: 0.5em;">Training a network and watching it learn on a test slice in real time.</p>

## Exporting a training dataset

After annotating, you first extract a training dataset. In the GUI this is **Create a training set** in the Models tab; on the command line it is [`ais extract`](../cli/extract.md). A training dataset is saved as a `.scnt` file. In older versions (before July 2026), this was really just a .tiff stack with some extra metadata. In current versions, `.scnt` files are uncompressed .tar archives that contain separate .mrc files for each of the training inputs and outputs.

You again specify a **box size** at this point. If you extract a larger box than you annotated, the un-annotated excess around the edges is given an *ignore* label during training — this is fine, nothing to worry about. If you extract smaller, a bit of your annotation effort simply goes to waste.

When exporting training data from the GUI, you can combine annotations for multiple features into one training set. Here you include each feature's annotations either as *positive examples* or as *negative examples*. See the explanation on the [annotation page](annotating.md) for more detail.

You can also set a **box depth**. Leave it at 1 and you'll train a 2D network. Increase it and you'll get a 2.5D or 3D network in the end:

- All networks can handle slabs, and train as **2.5D** networks by default — the depth dimension is treated as an extra channel, not as a spatial dimension.
- **Only some** networks turn slabs into true **3D** networks. These have `3d` in their name and can require a minimum slab depth: `ezm-3d-17d`, for example, is a 3D network that needs a slab of at least 17 slices.

In most cases you'll want to start with a simple 2D network.

!!! tip
    Exporting from the command line is a bit more powerful — it's faster and has some extra options. Really, all you need the GUI for is annotation and playing around: **export, training, batch segmentation, and picking are all much faster from the terminal.**

## Choosing an architecture

Ais comes with a library of about 15 different networks, and you can also [install your own architectures](../features/custom_architectures.md). For routine use you don't need to think about this too much:

- For an **initial test**, use the default — **VGGNet M**.
- Once you have a more thorough training dataset, it's worth trying a **larger model**, such as VGGNet L or UNet L.
- The most versatile and largest models are the **`ezm-2d-*`** family; the different variants use different loss functions. If your performance with the L models is disappointing despite a moderately sized training dataset (say, more than ~250 boxes total), try one of these — they're what we reach for once a good training dataset has been compiled.

Model choice, training-set size, and the rest are all a bit fuzzy. What matters is to keep your goal in mind and check whether your current model gets you there. When it doesn't, treat switching to a different model as an experiment: try something, assess the result, and draw a conclusion.

## Watching a network learn

When you train in the GUI, the network's output on a test slice updates as training progresses (you do have to force an update, by going to the next slice, or activating 'crop' in the Filters panel and dragging the ROI around a bit). The training loss is reported during training. If you want to use a validation split and see the validation loss instead, you can do so in Settings > Model settings > Validation splits. Don't read into the number too much — the best way to assess the output is to just look at it and see if you agree.

Training can be interrupted at any time, and you can continue training a saved model later by loading it (`-m` on the command line). When you're happy you can save the model, and if you're feeling generous you can also [share](../features/sharing_models.md) it via the [model repository](https://www.aiscryoet.org).

## Training parameters

The defaults are generally fine. In the training panel you can adjust the number of epochs, the batch size, and the number of augmented copies per input image used per epoch. See [`ais train`](../cli/train.md) for the full list of flags.

??? question "Epochs"
    How many times the network sees the whole dataset. ~25 is a good start. Too few and it underfits; too many and it may overfit — see [Assessing model performance](assessing_performance.md).

??? question "Batch size, copies & augmentation"
    Batch size (default 32) is usually fine. "Copies" sets how many augmented — rotated and flipped — versions of each positive patch are used; the default of 8 covers the eight 90° rotations and flips, and values larger than 8 add randomly rotated copies as well. On GPUs with less memory, or when using a large box size, you may need to reduce the batch size — GPU memory use scales roughly with batch size × box size².

??? question "Learning rate"
    The learning rate (default 1e-3, Adam optimizer) rarely needs changing; you can set it under Settings > Model settings > Learning rate. Lower it a little when you have a very large training dataset. You can also raise it to make training progress faster — but too high a rate can make training unstable and actually give worse results, so change it in small steps.
