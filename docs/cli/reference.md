# Command reference

Ais offers a command-line interface for the parts of the workflow that don't need the GUI: extracting training data, training, segmenting, and picking. These run faster from the terminal than from the GUI, and are easy to run on a cluster.

## `ais extract`

Extract training data from annotated tomograms. Annotations you save in the GUI are stored as `.scns` files; `ais extract` reads those and writes `.scnt` training-data files — one per feature — that `ais train` consumes. You set the box size, depth, and binning of the extracted data here.

```
ais extract -d <data_directory> -f <features...> [-ou <output_directory>] [-size <box_size>] [-depth <box_depth>] [-bin <binning>] [-e <exclude>] [-a <apix>] [--merge] [--coordinates]
```

### Options

| Option | Description |
| --- | --- |
| `-d`, `--data_directory` | Directory containing the annotated tomograms (`.scns` files). **Required.** |
| `-f`, `--features` | Features to extract, e.g. `-f Membrane Ribosome Microtubule`. A separate output file is written for each. **Required.** |
| `-ou`, `--output_directory` | Where to write the `.scnt` files. Default: current directory. |
| `-size`, `--box-size` | Box size in pixels. Default 128; if omitted, the size stored in the annotations is used. |
| `-depth`, `--box-depth` | Box depth in Z. Default 1 (2D). Must be odd (1 is added if not). Use a value >1 for a 2.5D dataset. |
| `-bin`, `--binning` | Binning factor in XY. The output box size becomes `box-size / binning`. Default 1. |
| `-e`, `--exclude` | Glob pattern, or a `.txt` file listing volumes to exclude from the dataset. |
| `-a`, `--apix` | Override the pixel size in the tomogram header with this value. |
| `--merge` | Write a single file per feature, pooling all input volumes, instead of one file per volume. |
| `--coordinates` | Export just the box coordinates as a `.star` file, rather than the training images. |

### Examples

```
ais extract -d warp_tiltseries/reconstruction/denoised -f Membrane -ou training_data
ais extract -d warp_tiltseries/reconstruction/denoised -f Membrane Ribosome -depth 5 --merge -ou training_data
```

The first command writes `training_data/128x128x1_Membrane.scnt` (the filename is `<box>x<box>x<depth>_<feature>.scnt`). The second writes one merged file per feature, e.g. `128x128x5_Membrane.scnt` and `128x128x5_Ribosome.scnt`.

## `ais train`

Train a segmentation network on one or more training datasets (`.scnt` files from [`ais extract`](#ais-extract)). The trained model is saved as an `.scnm` file.

```
ais train -t <training_data...> -ou <output_directory> -gpu <gpu_ids> -a <architecture> [-name <model_name>] [-e <epochs>] [-b <batch_size>] [-c <copies>] [-r <rate>] [-m <model_path>] [-augment] [-models]
```

### Options

| Option | Description |
| --- | --- |
| `-t`, `--training_data` | Path(s) to the `.scnt` file(s). Several may be given (`-t a.scnt b.scnt`); their samples are pooled. All must share the same box size and depth. |
| `-ou`, `--output_directory` | Directory to save the model in. Default: current directory. |
| `-gpu`, `--gpus` | Comma-separated GPU IDs, e.g. `0,1,2,3`. Default `0`. |
| `-a`, `--model_architecture` | Which architecture to train — its index **or** its title, e.g. `13` or `'VGGNet M'`. Run `ais train -models` for the list. |
| `-name`, `--model_name` | Model name. Saved as `output_directory/{name}.scnm`. |
| `-e`, `--epochs` | Number of epochs. Default 50. |
| `-b`, `--batch_size` | Batch size. Default 32. |
| `-c`, `--copies` | Augmented copies of each input image, in different orientations. Default 8 (the eight 90°-rotation + flip permutations); a value >8 adds randomly rotated copies; for 2.5D data, 8–16 also flip in Z. |
| `-r`, `--rate` | Learning rate. Default 1e-3 (Adam). |
| `-m`, `--model_path` | Continue training from a saved `.scnm`. Overrides `-a`. |
| `-augment` | Add extra augmentations (see the note below). |
| `-models`, `--model_architectures` | List the available architectures and their `-a` indices, then exit. |

!!! note "`-augment`"
    `-augment` adds scaling, contrast, brightness, and blurring augmentations on top of the default orientation augmentations. They are off by default, and you don't need them — but for a model that generalises a little better they can help. It takes some experimentation: train once without and once with, compare the two on your data, and keep whichever does better.

### Examples

List the architectures and their indices:

```
$ ais train -models
index: 0 (-a 0)    ezm-2d-dice
index: 1 (-a 1)    ezm-2d-bxe
...
index: 13 (-a 13)  VGGNet M
index: 14 (-a 14)  VGGNet S
...
```

Simplest case — point at the training data, name it, pick an architecture and GPUs:

```
ais train -t training_data/128x128x1_Membrane.scnt -name Membrane -ou models -gpu 0,1,2,3 -a 'VGGNet M'
```

With more settings — a lower learning rate, more epochs, and the extra augmentations:

```
ais train -t training_data/128x128x1_Membrane.scnt -name Membrane -ou models -gpu 0,1,2,3 -a 13 -e 100 -r 1e-4 -augment
```

## `ais segment`

Apply a trained model (`.scnm`) to segment `.mrc` volumes without the GUI.

```
ais segment -m <model_path> -d <data...> -ou <output_directory> -gpu <gpu_ids> [-tta <n>] [-overwrite <0|1>] [-apix <apix>] [-sigma <z y x>] [--batch <n>] [--workers <n>] [--center <percent>]
```

### Options

| Option | Description |
| --- | --- |
| `-m`, `--model_path` | Path to the model file (`.scnm`). **Required.** |
| `-d`, `--data` | One or more directories, files, or glob patterns for `.mrc` files, e.g. `/data/volumes`, `volumes/035*.mrc volumes/036*.mrc`. **Required.** |
| `-ou`, `--output_directory` | Directory to save the output. **Required.** |
| `-gpu`, `--gpus` | Comma-separated GPU IDs, e.g. `0,1,3,4`. **Required.** |
| `-tta`, `--test-time-augmentation` | Integer 1–8. If 1 (default), no test-time augmentation. If 2–8, differently oriented copies of the input are segmented and averaged (`[0, 90, 180, 270, 0*, 90*, 180*, 270*]`, `*` = horizontal flip). |
| `-overwrite` | If `1`, skip tomograms that already have a segmentation in the output directory. Default 0. See [Running on multiple nodes](#running-on-multiple-nodes). |
| `-apix`, `--processing_apix` | Process at this pixel size (Å/px) instead of the model's trained scale. Only needed when the pixel size in the `.mrc` header is wrong. |
| `-sigma`, `--postprocessing-blur-sigma` | Gaussian blur applied to the output (Å); one value or three (z y x). Off by default, and usually best left off — you can blur later without re-segmenting. |
| `--batch` | Slices per inference call. Default 1. You rarely need to change this. |
| `--workers` | CPU worker threads per GPU. Default: `cpu_count / n_gpus`. You rarely need to change this. |
| `--center` | Percentage of the volume depth (Z) to segment, centred on the middle. E.g. `--center 50` segments only the central half. Default 100. |

### Running on multiple nodes

You can launch several `ais segment` commands against the same output directory at once — even from different nodes — and the work is distributed automatically. Each GPU claims a tomogram by writing a small placeholder `.mrc` before it starts, so no two processes pick up the same volume, and `-overwrite 0` (the default) means volumes that already have an output are skipped.

If a process crashes, its placeholder can be left behind. The tomogram then looks segmented — there is an output file — but the file is tiny and empty. Delete such placeholders by hand and re-run to segment those volumes properly.

### Output filenames

Ais, Pom, and easymode name a segmentation `<tomogram>__<model>.mrc` (double underscore) — for a tomogram `tomo_001.mrc` segmented with a model titled `ribosome`, the output is `tomo_001__ribosome.mrc`. The Render tab uses this convention to find the segmentations belonging to a tomogram automatically.

### Examples

```
ais segment -m models/Membrane.scnm -d warp_tiltseries/reconstruction/denoised -ou segmentations -gpu 0,1,2,3,4,5,6,7
ais segment -m models/Microtubule.scnm -d "warp_tiltseries/reconstruction/denoised/TS_001*.mrc" -ou segmentations -gpu 0,1 -tta 4 -overwrite 1
```

## `ais pick`

Turn segmented volumes into particle coordinates. `ais pick` reads the segmentation `.mrc` files produced by [`ais segment`](#ais-segment) and writes RELION-style `.star` coordinate files. It has two modes: **blob** (the default, for compact particles) and **filament** (`-filament`, for filaments such as microtubules).

```
ais pick -d <data_directory> -t <target> [-ou <output_directory>] [-threshold <v>] [-spacing <A>] [-size <A^3>] [-m <margin>] [-b <binning>] [-p <n>] [--subset <file>] [blob / filament options]
```

### Options

| Option | Description |
| --- | --- |
| `-d`, `--data_directory` | Directory of input segmentation `.mrc` files. **Required.** |
| `-t`, `--target` | Feature to pick. For volumes named `<tomo>__Ribosome.mrc`, `-t Ribosome` selects them. **Required.** |
| `-ou`, `--output_directory` | Where to save the `.star` files. Default: the input directory. |
| `-threshold` | Threshold applied before finding maxima. Default 128 (Ais volumes are 0–255). |
| `-spacing`, `-spacing-px` | Minimum distance between particles, in Ångström (`-spacing`) or voxels (`-spacing-px`). |
| `-size`, `-size-px` | Minimum particle size, in cubic Ångström (`-size`) or voxels (`-size-px`). |
| `-m`, `--margin` | Margin (px) to avoid picking near tomogram edges. Default 16. |
| `-b`, `--binning` | Binning applied before processing (faster, less precise). Default 1. |
| `-min-particles` | Minimum particles a tomogram must yield for its `.star` file to be written. Default 0. |
| `-p`, `--parallel` | Number of parallel picking processes, e.g. `-p 64`. Default 1. |
| `--subset` | A `.txt` file listing tomogram names, one per line (e.g. a [Pom](../features/pom.md) subset). Only matching volumes are picked. |
| `-v`, `--verbose` | Verbose output (`1` or `0`). Default 0. |

#### Blob mode (default)

Each connected component becomes one coordinate, placed at its deepest (highest-value) point.

| Option | Description |
| --- | --- |
| `-centroid` | Place coordinates at each component's centroid instead of its deepest point. Use only when particles are well separated. |
| `-orient` | With `-centroid`, also write Euler angles from each blob's shape: `normal` (disk normal / smallest principal axis) or `long-axis` (rod axis / largest principal axis). Sets `rlnAngleTilt` and `rlnAnglePsi`. |
| `-orient-sign` | How to resolve the axis sign: `z` (force +z, default), `center`, or `out`. |

#### Filament mode (`-filament`)

Coordinates are placed evenly along filaments rather than at blobs.

| Option | Description |
| --- | --- |
| `-length`, `-length-px` | Minimum filament length to place coordinates along, in Ångström (`-length`) or pixels (`-length-px`). Default 500 Å. |
| `--twist` | Increment `rlnAngleRot` by this amount for each particle along a filament. |

### Examples

```
ais pick -d segmentations -t Ribosome -ou coordinates -threshold 128 -spacing 250 -size 1000000 -p 64
ais pick -d segmentations -t Microtubule -ou coordinates -filament -length 800 -spacing 82
```

When picking from Ais volumes, values are 0–255 and 128 is a good default threshold. Use the Ais 3D renderer to find the threshold, spacing, and size values that work for your target.
