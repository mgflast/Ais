# Assessing model performance

The main loop in the Ais workflow is: annotate, train, test. You do this for a bit — 2, maybe 3 iterations — until you land on a network that does well on your data. But how do you check whether your network works well? We explain that here.

## Testing a model on just a few tomograms

The best way to test a model is to load it into the GUI and apply it to some (unseen) tomograms. If you trained in the GUI, the model will already be there. If you trained via the CLI, drag & drop the model (`.scnm`) file into the editor, or open it via File → Load model in the top menu bar.

For every pixel in a slice the model outputs a value between 0.0 and 1.0. Because the training labels are binary, these values mostly cluster around 0 or 1. To see a bit more variation, use **test-time augmentation** (Settings → Model settings → TTA multiplicity): the model processes each slice several times, in different orientations and flips, and averages the results. TTA is available on the command line too (`ais segment ... -tta 4`). Try setting the TTA multiplicity to 4, navigate to a fresh slice, and play with the threshold value. More TTA is more better but also slower.

Inspecting a model's output and improving it are the same loop: wherever the model is wrong is exactly where an extra annotation will help most. [Model-assisted annotation](../features/model_assisted_annotation.md) makes this fast — copy the parts the model got right into your training data, fix the rest, and train again.

Testing on previously unseen tomograms is the most useful: when the model makes a mistake there, you can add boxes from that new tomogram to improve the training data. Sampling broadly is generally more useful than placing a lot of boxes in a single tomogram — different tomograms vary in defocus, thickness, alignment quality, and content, so including several of them is good for generalisation.

<video controls muted loop autoplay playsinline width="100%">
  <source src="../../res/testing_a_model.mp4" type="video/mp4">
</video>
<p style="text-align: center; font-style: italic; color: var(--md-default-fg-color--light); margin-top: 0.5em;">After training a model to segment membranes, we test it on a small set of tomograms and use model-assisted annotation to quickly generate additional training samples, for those regions where the model makes mistakes.</p>

## Testing a model on an entire dataset

When your model seems mostly good, it's useful to test it against your entire dataset. That's too much for the GUI — instead, run [`ais segment`](../cli/reference.md#ais-segment) first:

```
ais segment -m my_model.scnm -d my_tomogram_dir/ -ou test_segmentations/ -gpu 0,1,2,3 -tta 4
```

Then use [Pom](https://mgflast.github.io/easymode/user_guide/pom/data_browser/) to visualise and explore all of the segmentation results. If you [synchronise Pom and Ais](../features/pom.md), you can use Pom to open tomograms — or previously annotated `.scns` files — directly in Ais. 
