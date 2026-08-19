# Quick start

In this tutorial we will run through the basics of creating a model in Ais. After following this tutorial, you will know your way around the GUI and be able to follow along easily with any other tutorials, or you can get started with your own data. Completing all the steps should take only about 15 to 30 minutes.
!!! note
    Every section of this tutorial ends with a video that shows the steps being done in Ais. The videos include an overlay that displays the keyboard and mouse input.

## 1. Getting the data & launching Ais

We'll use a tomogram from [EMPIAR-13566](https://www.ebi.ac.uk/empiar/EMPIAR-13566/) — primary human T cells imaged <i>in situ</i>, denoised with IsoNet2, uploaded by Jan Philipp Kreysing, Martin Beck, and colleagues. Activate your `ais` environment and copy and run the below script.

```
import os
import urllib.request
import mrcfile
from tqdm import tqdm

URL = "https://ftp.ebi.ac.uk/empiar/world_availability/13566/data/bin4_rec/019_IsoNet2.mrc"
OUT = "tomograms/019_IsoNet2.mrc"
APIX = 7.58

os.makedirs("tomograms", exist_ok=True)
if not os.path.exists(OUT):
    with urllib.request.urlopen(URL) as r, open(OUT, "wb") as f:
        with tqdm(total=int(r.headers["Content-Length"]), unit="B", unit_scale=True) as bar:
            while chunk := r.read(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))

with mrcfile.mmap(OUT, mode="r+") as m:
    m.voxel_size = APIX     # Ais assumes the voxel size is correct, and scales by it during training and inference
```

Once the download completes, launch Ais with the command `ais`, then open the tomogram by dragging & dropping the file into Ais, or in the menu bar `File > Import datasets` and selecting it there.

<video controls muted loop autoplay playsinline width="100%">
  <source src="../../res/quick_start_1.mp4" type="video/mp4">
</video>
<p style="text-align: center; font-style: italic; color: var(--md-default-fg-color--light); margin-top: 0.5em;">Launching Ais and importing the tomogram.</p>

## 2. Setting up our features of interest
Lots of different features are visible in the tomogram: nucleosomes, proteasomes, TRiC particles, intermediate filaments, ribosomes, a nuclear pore complex, and actin, to name a few. For the tutorial we'll do membranes.

In the annotation tab, click Add feature and in the drop-down menu go to Feature library > open feature library. Here we will set up defaults for our features of interest. 

Add a new feature, name it `membrane` and assign it a colour, then click save and close the Feature library window. Now in the drop-down menu of your annotation, your preset values for membrane annotations are there. Select it.

<video controls muted loop autoplay playsinline width="100%">
  <source src="../../res/quick_start_2.mp4" type="video/mp4">
</video>
<p style="text-align: center; font-style: italic; color: var(--md-default-fg-color--light); margin-top: 0.5em;">Setting up a membrane feature in the feature library.</p>

## 3. Drawing some membrane training data
We will just annotate a couple of boxes, then use those to train a small model. Select the membrane panel to activate it. By holding `shift`, the cursor goes into `box` mode. Wherever you place a box is what data gets sent to the models. Place a box, and annotate membranes within it. Draw with `left mouse button`, erase with `right mouse button`. You can change the size of the brush with `ctrl` + scrolling the mouse wheel, or by holding `ctrl` and the middle mouse button and moving the cursor left and right.

Besides annotating a bit of membrane, it also helps to include examples of what is _not_ membrane. For the same membrane annotation, place some more boxes in areas that do not contain membranes.

Once you have 5 boxes with annotated membrane and around 10 with background, press `ctrl` + `S` to save the annotations.

<video controls muted loop autoplay playsinline width="100%">
  <source src="../../res/quick_start_3.mp4" type="video/mp4">
</video>
<p style="text-align: center; font-style: italic; color: var(--md-default-fg-color--light); margin-top: 0.5em;">Placing boxes and annotating membranes.</p>

## 4. Training an initial model
That's the annotation step completed. Now go to the second tab 'Models' (you can press `1` and `2` to toggle between the annotation and models tabs) and open the 'Create a training set' panel. Select membrane as your feature of interest by dragging its selector to the right, and change the set parameters to a box size of 128, box depth of 1, and a pixel size of 7.6 A/px. Click 'Generate set' and save the training data to the project directory.

Now click 'Add model' and using the feature library dropdown, apply your preset title & colour to this new model. In the Training sub-tab, click 'browse', select the training data (.scnt) file you just saved, and finally press 'train' to start training a model with the default settings. On our desktop (RTX A1000) this took about a minute.

<video controls muted loop autoplay playsinline width="100%">
  <source src="../../res/quick_start_4.mp4" type="video/mp4">
</video>
<p style="text-align: center; font-style: italic; color: var(--md-default-fg-color--light); margin-top: 0.5em;">Generating a training set and training a first model.</p>

## 5. Using the model to improve the training data.
With a model loaded in the GUI, you can see its output as your scroll through the tomogram. In the model panel's `Prediction` tab, you can adjust the binarization threshold and the test-time augmentation fold (TTA). Higher values tend to improve the output, but slow things down as well. Test-time augmentation means that the network processes a slice multiple times in different orientations (rotated 0, 90, 180, 270 degrees, and optionally also mirrored - giving a maximum 8-fold augmentation). This helps to average out errors. 

Because we only did a tiny bit of annotation, the model will probably not be very accurate. Find a slice where the output is wrong, then use the `model-assisted annotation` tool to copy the output over into the annotations tab to edit them. Then place some more boxes to include additional examples of membranes in the training data.

Even when a model isn't great, model-assisted annotation can help you speed up annotation & identify model biases. Using it, try to add 10 more annotated boxes and 20 more containing just background. Then repeat the training procedure from before: go to the Models tab, export the training data (same settings as before), and continue training the model.

<video controls muted loop autoplay playsinline width="100%">
  <source src="../../res/quick_start_5.mp4" type="video/mp4">
</video>
<p style="text-align: center; font-style: italic; color: var(--md-default-fg-color--light); margin-top: 0.5em;">Using model-assisted annotation to correct and extend the training data.</p> 

## 6. Exporting a segmentation
Once you're happy with the model, go to the next tab: `Export` (or press key `3`). In general it is faster and more versatile to do your segmentations from the command line, but we will do one in the GUI for now. For large datasets, use `easymode segment` on a GPU node. This requires a model saved to disk. In in model tab, for the membrane model panel, go to `Training > Save` to save the model for later re-use.

In the video below we set up the export on the `Export` tab: selecting which tomograms to process (only one in this case), choosing to limit the range of slices to process (just for it to be a bit faster), and setting where to save the output to. Again, in general, we recommend doing segmentation via the cli instead.

<video controls muted loop autoplay playsinline width="100%">
  <source src="../../res/quick_start_6.mp4" type="video/mp4">
</video>
<p style="text-align: center; font-style: italic; color: var(--md-default-fg-color--light); margin-top: 0.5em;">Setting up batch segmentation in the Export tab.</p>

For us the processing took about 60 seconds (TTA set back to 1). The equivalent command for this segmentation would have been: `easymode segment -m membrane.scnm -d tomograms/019_IsoNet2.mrc -ou segmented/ -gpu 0 -tta 1`.

## 7. Visualizing the results in 3D
When the segmentation is completed, you can visualize the results in the `Render` tab (key `4`). Ais will automatically link segmentation files to their corresponding tomograms. Select the directory where you saved the segmentation and the membrane segmentation volume should appear - if not, use the manual import button.

<video controls muted loop autoplay playsinline width="100%">
  <source src="../../res/quick_start_7.mp4" type="video/mp4">
</video>
<p style="text-align: center; font-style: italic; color: var(--md-default-fg-color--light); margin-top: 0.5em;">Inspecting the segmentation as a 3D render in the Render tab.</p>

At this point you can start thinking about particle picking. If the segmentations look good in 3D, you can run `ais pick` on them rightaway; alternatively, if the 3D visualizations still show errors, you can improve the network by doing another iteration of (model-assisted) annotation and training. For membranes as a whole there isn't much to pick, so we end this tutorial here. If you are interested in using Ais for segmentation-based picking, see the [tutorial on picking microtubule inner proteins](microtubule_inner_proteins.md).