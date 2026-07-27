# Command reference

Ais offers a command-line interface for the parts of the workflow that don't need the GUI: extracting training data, training, segmenting, and picking. Each of these is faster from the terminal than from the GUI, and easy to run on a cluster.

## `ais extract`

Extract training data (`.scnt` files) from annotated tomograms (`.scns` files). This is the first step of the command-line workflow: turn the annotations you drew in the Ais GUI into training datasets that `ais train` can consume. A separate output file is created for each feature.

```
ais extract -d <data_directory> -f <features...> [-ou <output_directory>] [-size <box_size>] [-depth <box_depth>] [-bin <binning>] [-e <exclude>] [-a <apix>] [--merge] [--coordinates]
```

### Options

| Option | Description |
| --- | --- |
| `-d`, `--data_directory` | Directory containing annotated tomograms (`.scns` files). **Required.** |
| `-f`, `--features` | List of features to extract, e.g. `-f Membrane Ribosome Microtubule`. A separate output file is created for each feature. **Required.** |
| `-ou`, `--output_directory` | Directory to save the extracted training data (`.scnt` files). Default: current directory. |
| `-size`, `--box-size` | Box size (in pixels) to extract. When not specified, the box size is taken from the annotations. Default 128. |
| `-depth`, `--box-depth` | Box depth (in Z) to extract. Default 1 (2D). Must be odd — if not odd, 1 is added. Use a value >1 for a 2.5D dataset. |
| `-bin`, `--binning` | Binning factor to apply (in XY). Output box size will be `--box-size / --binning`. Default 1. |
| `-e`, `--exclude` | Glob pattern or path to a `.txt` file listing volumes to exclude from the extracted dataset. |
| `-a`, `--apix` | Override the pixel size found in the tomogram header and use this value instead. |
| `--merge` | Combine all extracted data into a single output file per feature, rather than one file per input volume. |
| `--coordinates` | Instead of exporting annotated training images, export just the box coordinates as a `.star` file. |

### Examples

```
ais extract -d annotations -f Microtubule -ou training_data -size 64
ais extract -d annotations -f Membrane Ribosome -ou training_data -depth 5 --merge
```

## `ais train`

Train a segmentation network on one or more training datasets (`.scnt` files, produced by [`ais extract`](#ais-extract)). The trained model is saved as an `.scnm` file.

```
ais train -t <training_data...> -ou <output_directory> -gpu <gpu_ids> [-a <architecture>] [-m <model_path>] [-e <epochs>] [-b <batch_size>] [-c <copies>] [-r <rate>] [-n <negatives>] [-p <parallel>] [-augment] [-name <model_name>] [-models]
```

### Options

| Option | Description |
| --- | --- |
| `-t`, `--training_data` | Path(s) to the training data (`.scnt`) file(s). Multiple files may be given, e.g. `-t a.scnt b.scnt`; their samples are pooled during training. All files must share the same box size and depth. |
| `-ou`, `--output_directory` | Directory to save the model in. Default: current directory. |
| `-gpu`, `--gpus` | Comma-separated list of GPU IDs to use, e.g. `0,1,4,5`. Default `0`. |
| `-a`, `--model_architecture` | Integer index of which model architecture to use. Use `-models` for a list of available architectures. |
| `-m`, `--model_path` | Path to a previously saved model (`.scnm`) to continue training. Overrides `-a`. |
| `-e`, `--epochs` | Number of epochs to train for. Default 50. |
| `-b`, `--batch_size` | Batch size to use during training. Default 32. |
| `-c`, `--copies` | Number of augmented versions of each input image to include (all in different orientations). Default 8 (the eight permutations of 90° rotations + horizontal flips). A value >8 adds randomly rotated versions. For 2.5D data, values 8–16 also include a flip in Z. |
| `-r`, `--rate` | Learning rate. Default 1e-3. All default Ais networks use Adam as the optimizer. |
| `-n`, `--negatives` | If 0.0 (default), all images are weighted identically. Otherwise, sets the ratio of negative to positive samples (some negatives are sampled more than once to reach the ratio). |
| `-p`, `--parallel` | `1` (default) or `0`: whether to use TensorFlow's `distribute.MirroredStrategy()` across multiple GPUs, or a single process using all GPUs. |
| `-augment` | If set, use extra scaling, contrast, brightness, and blurring augmentations. |
| `-name`, `--model_name` | Model name. File is saved as `output_directory/{name}.scnm`. |
| `-models`, `--model_architectures` | List available model architectures and their `-a` indices, then exit. |

### Examples

```
ais train -models
ais train -t training_data/Microtubule.scnt -ou models -gpu 0,1,2,3 -a 5 -e 25 -name Microtubule
ais train -m models/Microtubule.scnm -t training_data/Microtubule_extra.scnt -ou models -gpu 0 -name Microtubule
```

## `ais segment`

Apply a trained model (`.scnm`) to segment `.mrc` volumes without the GUI. Output segmentation `.mrc` files have the same shape as the input tomograms.

```
ais segment -m <model_path> -d <data...> -ou <output_directory> -gpu <gpu_ids> [-tta <n>] [-p <parallel>] [-overwrite <0|1>] [-apix <apix>] [-sigma <z y x>] [--batch <n>] [--workers <n>] [--center <percent>]
```

### Options

| Option | Description |
| --- | --- |
| `-m`, `--model_path` | Path to the model file (`.scnm`). **Required.** |
| `-d`, `--data` | One or more directories, file paths, or glob patterns for `.mrc` files. Examples: `/data/volumes`, `volumes/035*.mrc volumes/036*.mrc`, or explicit files. **Required.** |
| `-ou`, `--output_directory` | Directory to save the output. **Required.** |
| `-gpu`, `--gpus` | Comma-separated list of GPU IDs, e.g. `0,1,3,4`. **Required.** |
| `-tta`, `--test-time-augmentation` | Integer 1–8. If 1 (default), no test-time augmentation. If 2–8, differently oriented copies of the input are segmented and averaged; orientations are `[0, 90, 180, 270, 0*, 90*, 180*, 270*]` (`*` = horizontal flip). |
| `-p`, `--parallel` | `1` (default) or `0`: launch multiple parallel processes using one GPU each, or a single process using all GPUs. |
| `-overwrite` | If `1`, tomograms with an existing segmentation in the output directory are skipped. Default 0. |
| `-apix`, `--processing_apix` | Override the model's trained scale (Å/px) and process at this value. |
| `-sigma`, `--postprocessing-blur-sigma` | Gaussian postprocessing sigma (Å): a single value (isotropic) or three (z y x). Default: no blur. |
| `--batch` | Number of slices to batch per inference call. Default 1. Increase for faster inference if GPU memory allows. |
| `--workers` | CPU worker threads per GPU for pre/postprocessing. Default: `cpu_count / n_gpus`. |
| `--center` | Percentage of the volume depth (Z) to segment, centred on the middle. E.g. `--center 50` segments only the central 50%. Default 100. |

### Multi-GPU segmentation

On systems with multiple GPUs, segmenting several volumes at once with parallel single-GPU processes (`-p 1`) is usually much faster than one process using all GPUs.

You can also launch several `ais segment` commands against the same directory concurrently — the workload is distributed automatically, so every process contributes to the same job. Occasionally a few tiny placeholder `.mrc` files (e.g. 10×10×10, a few kB) are left behind; these can be safely deleted.

### Output filenames

Ais, Pom, and easymode use a fixed filename convention to link segmentations back to their tomograms. For a tomogram `tomo_001.mrc` segmented with a model titled `ribosome`, the output is `tomo_001__ribosome.mrc` — that is, `<tomogram>` + `__` (double underscore) + `<model>` + `.mrc`. Thanks to this, the Render tab in the GUI automatically finds and displays the segmentations belonging to a tomogram.

### Examples

```
ais segment -m models/Membrane.scnm -d volumes -ou segmentations -gpu 0,1,2,3,4,5,6,7
ais segment -m models/Microtubule.scnm -d "volumes/TS_001*.mrc" -ou segmentations -gpu 0,1 -tta 4 -overwrite 1
```

## `ais pick`

Turn segmented volumes into particle coordinates. `ais pick` finds local maxima in the segmentation `.mrc` files produced by [`ais segment`](#ais-segment) and writes RELION-style `.star` coordinate files.

```
ais pick -d <data_directory> -t <target> [-ou <output_directory>] [-threshold <v>] [-spacing <A>] [-size <A^3>] [-m <margin>] [-b <binning>] [-p <parallel>] [-filament] [-centroid] [-orient <mode>] ...
```

### Options

| Option | Description |
| --- | --- |
| `-d`, `--data_directory` | Directory containing input segmentation `.mrc` files, e.g. `/segmented/`. **Required.** |
| `-t`, `--target` | Feature to pick. If segmented volumes are named `<tomo>__Ribosome.mrc`, then `-t Ribosome` selects them. **Required.** |
| `-ou`, `--output_directory` | Directory to save coordinate files. If empty, saves to the input directory. |
| `-threshold` | Threshold applied to volumes before finding local maxima. Default 128. |
| `-spacing` / `-spacing-px` | Minimum distance between particles, in Ångström (`-spacing`) or voxels (`-spacing-px`). |
| `-size` / `-size-px` | Minimum particle size, in cubic Ångström (`-size`) or voxels (`-size-px`). |
| `-m`, `--margin` | Margin (px) to avoid picking near tomogram edges. Default 16. |
| `-b`, `--binning` | Binning factor applied before processing (faster, possibly less accurate). Default 1. |
| `-min-particles` | Minimum number of particles required in a tomogram for the `.star` file to be saved. Default 0. |
| `-p`, `--parallel` | Number of parallel picking processes, e.g. `-p 64`. Default 1. |

#### Blob vs filament mode

| Option | Description |
| --- | --- |
| `-centroid` | Blob mode: place coordinates at the centroid of each connected component rather than the deepest point. Only use when particles are well separated. |
| `-orient` | In centroid mode, also write Euler angles from each blob's shape: `normal` (disk normal / smallest principal axis) or `long-axis` (rod axis / largest principal axis). Sets `rlnAngleTilt` and `rlnAnglePsi`. |
| `-orient-sign` | Convention for resolving the axis sign: `z` (force +z, default), `center`, or `out`. |
| `-filament` | Pick in filament mode rather than blob mode. |
| `--twist` | In filament mode, increment `rlnAngleRot` by this amount for each particle along a filament. |
| `-length` / `-length-px` | In filament mode, minimum filament length to place coordinates along, in Ångström or pixels. Default 500 Å. |

#### Pom integration

| Option | Description |
| --- | --- |
| `-capp`, `--pom-capp-config` | A [Pom](https://github.com/mgflast/Pom) context-aware particle-picking configuration file (optional). |
| `--subset` | Path to a `.txt` file listing tomogram names/paths (e.g. a Pom subset file). Only matching volumes are picked. |

When picking coordinates from Ais-generated volumes, voxel values are 0–255 and 128 is a good default threshold. Use the Ais 3D renderer to test which threshold, spacing, and size values work well for your target.

### Examples

```
ais pick -d segmentations -t Ribosome -ou coordinates -threshold 128 -spacing 250 -size 1000000 -p 64
ais pick -d segmentations -t Microtubule -ou coordinates -filament -length 800 -spacing 82
```
