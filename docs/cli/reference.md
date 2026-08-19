# Command reference

Ais offers a command-line interface for the parts of the workflow that don't need the GUI: extracting training data, training, segmenting, and picking. These run faster from the terminal than from the GUI, and are easy to run on a cluster.

## `ais extract`

Extract training data from annotated tomograms. Annotations you save in the GUI are stored as `.scns` files; `ais extract` reads those and writes `.scnt` training-data files — one per feature — that `ais train` consumes. Extracting can be done from the GUI, but if you have `.scns` files on disk that you want to extract your labelled training data from, you don't have to open the GUI at all. Important parameters are `-size`, `-depth`, and `-apix`.

```
ais extract -d <data_directory> -f <features...> [-ou <output_directory>] [-size <box_size>] [-depth <box_depth>] [-e <exclude>] [-a <apix>] [--merge] [--coordinates]
```

### Options

| Option | Description |
| --- | --- |
| `-d`, `--data_directory` | Directory containing the annotated tomograms (`.scns` files). **Required.** |
| `-f`, `--features` | Features to extract, e.g. `-f Membrane Ribosome Microtubule`. A separate output file is written for each. **Required.** |
| `-ou`, `--output_directory` | Where to write the `.scnt` files. Default: current directory. |
| `-size`, `--box-size` | Box size in pixels. Default 128; if omitted, the size stored in the annotations is used. |
| `-depth`, `--box-depth` | Box depth in Z. Default 1 (2D). Must be odd (1 is added if not). Use a value >1 for a 2.5D dataset. |
| `-e`, `--exclude` | Glob pattern, or a `.txt` file listing volumes to exclude from the dataset. |
| `-a`, `--apix` | Target pixel size for the extracted boxes (default 10.0). Together with `-size`, this determines the actual field of view size in any extracted box. |
| `--merge` | Write a single file per feature, pooling all input volumes, instead of one file per volume. |
| `--coordinates` | Export just the box coordinates as a `.star` file, rather than the training images. |

### Examples

```
ais extract -d warp_tiltseries/reconstruction/denoised -f Membrane -ou training_data
ais extract -d warp_tiltseries/reconstruction/denoised -f Membrane Ribosome -depth 5 --merge -ou training_data
```

The first command writes `training_data/128x128x1_Membrane.scnt` (the filename is `<box>x<box>x<depth>_<feature>.scnt`). The second writes one merged file per feature, e.g. `128x128x5_Membrane.scnt` and `128x128x5_Ribosome.scnt`. 

*Note:* it is okay for the `-size` and `-apix` arguments to combine into boxes larger than those you annotated. Any image region not contained within a box (a box as placed during annotation in the Ais GUI) will have an 'ignore label' written, meaning it does not factor in to the training. Even though the loss is not scored on these image areas, they still do provide more spatial context during training, which can be helpful.

??? note "What's in a .scnt file?"

    An .scnt file is essentially just a .tar archive that contains two separate directories, one for the training input and one for the training output: `x_main/` and `y/`. Within these directories are .mrc files, one for each training sample. Between the different directories, files are linked by their name; e.g. `x_main/e2d104fa.mrc` is the training input for output `y/e2d104fa.mrc`. Besides the actual data, a `.scnt` file also contains a `metadata.json` file.

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
| `-a`, `--model_architecture` | Which architecture to train — its index or its title, e.g. `13` or `'VGGNet M'`. Run `ais train -models` to print a list of available architectures. |
| `-name`, `--model_name` | Model name. Saved as `output_directory/{name}.scnm`. |
| `-e`, `--epochs` | Number of epochs. Default 50. |
| `-b`, `--batch_size` | Batch size. Default 32. |
| `-c`, `--copies` | Number of augmented copies of each input image served per epoch. Augmentations include rotations (0, 90, 180, 270 degrees) (for `-c 1` to `-c 4`), flipped and rotated copies (`-c 5` to `-c 8`), and copies randomly rotated around the Z axis (`-c >8`). For 2.5D or 3D data, `-c 9` to `-c 16` additionally include a rotation around the X axis (and not the random rotation around Z). |
| `-r`, `--rate` | Learning rate. Default 1e-3. |
| `-m`, `--model_path` | Continue training from a saved `.scnm`. Overrides `-a`, because the saved weights only work for the architecture they represent. |
| `--filament` | Filament tube diameter (px). For the 3D filament architectures (e.g. `ezm-3d-filament`) only. |
| `-augment` | Add extra augmentations (see the note below). |
| `-models`, `--model_architectures` | List the available architectures and their `-a` indices, then exit. |

!!! note "`-augment`"
    `-augment` adds scaling, contrast, brightness, noise, blurring, and gamma augmentations on top of the default orientation augmentations. They are off by default, and you don't need them — but it can help models achieve a little better generalisation. Scaling is ±10% (in XY), contrast jitter ±10%, brightness jitter an offset of ±0.1, added Gaussian noise has σ = 0.2, blurring is a Gaussian filter with σ between 0.1 and 1.1, and gamma ranges 0.8–1.2.

### Examples

List the architectures and their indices:

```
$ ais train -models
index: 0 (-a 0)    ezm-2d-dice
index: 1 (-a 1)    ezm-2d-bxe
...
index: 12 (-a 12)  VGGNet M
index: 13 (-a 13)  VGGNet S
...
```

Simplest case — point at the training data, name it, pick an architecture and GPUs:

```
ais train -t training_data/128x128x1_Membrane.scnt -name Membrane -ou models -gpu 0,1,2,3 -a 'VGGNet M'
```

With more settings — a lower learning rate, more epochs, and the extra augmentations:

```
ais train -t training_data/128x128x1_Membrane.scnt -name Membrane -ou models -gpu 0,1,2,3 -a 12 -e 100 -r 1e-4 -augment
```

## `ais segment`

Apply a trained model (`.scnm`) to segment `.mrc` volumes without the GUI.

```
ais segment -m <model_path> -d <data...> -ou <output_directory> -gpu <gpu_ids> [-tta <n>] [-overwrite <0|1>] [-apix <apix>] [--batch <n>] [--workers <n>] [--center <percent>]
```

### Options

| Option | Description |
| --- | --- |
| `-m`, `--model_path` | Path to the model file (`.scnm`). **Required.** |
| `-d`, `--data` | One or more directories, files, or glob patterns for `.mrc` files, e.g. `/data/volumes`, `volumes/035*.mrc volumes/036*.mrc`. **Required.** |
| `-ou`, `--output_directory` | Directory to save the output. **Required.** |
| `-gpu`, `--gpus` | Comma-separated GPU IDs, e.g. `0,1,3,4`. **Required.** |
| `-tta`, `--test-time-augmentation` | Integer 1–8. If 1 (default), no test-time augmentation. If 2–8, differently oriented copies of the input are segmented and averaged (`[0, 90, 180, 270, 0*, 90*, 180*, 270*]`, `*` =  flip in X). |
| `-overwrite` | If `1`, tomograms that already have a segmentation in the output directory are re-segmented. Default `0`: they are skipped. See [Running on multiple nodes](#running-on-multiple-nodes). |
| `-apix`, `--processing_apix` | Process at this pixel size (Å/px) instead of the model's trained scale. |
| `-data-apix` | Override the pixel size in the input volumes' headers (Å/px). Use when the header value is missing or incorrect; `0.0` disables rescaling entirely. |
| `--batch` | Slices per inference call. Default 1. You rarely need to change this. |
| `--workers` | CPU worker threads per GPU. Default: `cpu_count / n_gpus`. You rarely need to change this. |
| `--center` | Percentage of the volume depth (Z) to process, centred on the middle. E.g. `--center 50` segments only the central half. Default 100, reduce for speed. |

### Running on multiple nodes

You can launch several `ais segment` commands on the same input and output directory at once on different nodes and the work is distributed automatically. Each GPU claims a tomogram by writing a small placeholder `.mrc` before it starts, so no two processes pick up the same volume, and `-overwrite 0` (the default) means volumes that already have an output are skipped.

If a process crashes, its placeholder is sometimes left behind. The tomogram then looks segmented — there is an output file — but the file is tiny and empty. Delete such placeholders before re-running segmentation to make sure you get proper output for those tomograms.

### Output filenames

Ais, Pom, and easymode name a segmentation `<tomogram>__<model>.mrc` (double underscore) — for a tomogram `tomo_001.mrc` segmented with a model titled `ribosome`, the output is `tomo_001__ribosome.mrc`. The Render tab uses this convention to find the segmentations belonging to a tomogram automatically.

### Examples

```
ais segment -m models/membrane.scnm -d warp_tiltseries/reconstruction/denoised -ou segmentations -gpu 0,1,2,3,4,5,6,7
ais segment -m models/microtubule.scnm -d "warp_tiltseries/reconstruction/denoised/TS_01*.mrc" -ou segmentations -gpu 0,1 -tta 4 -overwrite 1
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

**Algorithm**: input segmentations are first thresholded. Connected components with a volume smaller than `-size` (in cubic Angstrom) are set to 0. Then a distance map is calculated. In this map each voxel's value corresponds to the distance to the nearest 0-valued voxel in the thresholded segmentation - i.e. how 'deep' inside of a connected nonzero component the voxel is. Coordinates are then placed at maxima in this distance map. After placing coordinates at all maxima, a minimum spacing is applied: for every pair of coordinates C0 and C1 within a distance D (`-spacing`) of each other, the coordinate with the lowest distance value is discarded.

**Optional** `-centroid` *mode*: instead of calculating a distance map, calculate the center of mass of every connected component. This works when connected components represent a single particle only. If this is not the case - for example, polyribosomes for which the segmentation is one connected blob - this mode significantly underpicks. The benefit of centroid mode is that based on the shape of the blob you can sometimes determine approximate particle poses. This assumes the blob is non-spherical. Nuclear pore complexes are an example where this is useful.

| Option | Description |
| --- | --- |
| `-centroid` | Place coordinates at each component's centroid instead of its deepest point. Use only when particles are well separated. |
| `-orient` | With `-centroid`, also write Euler angles from each blob's shape: `normal` (disk normal / smallest principal axis) or `long-axis` (rod axis / largest principal axis). Sets `rlnAngleTilt` and `rlnAnglePsi`. |
| `-orient-sign` | How to resolve the axis sign: `z` (force +z, default), `center`, or `out`. |

#### Filament mode (`-filament`)

**Algorithm**: segmentations are thresholded and small connected components (volume < `-size`) are set to 0. The binary volume is then skeletonized and branches with a contour length below `-length` are pruned. A spline is fitted to the remaining branches, and this spline is sampled in steps of `-spacing`, incrementing rlnAngleRot by `--twist` for every coordinate placed. Euler angles rlnAngleTilt and rlnAnglePsi are derived from the tangent to the spline. This offers a very useful prior for filaments. Note that the sign of the tangent is arbitrarily chosen, which means that for polar filaments the tangent may need to be flipped (on a per-filament basis).

| Option | Description |
| --- | --- |
| `-length`, `-length-px` | Minimum filament length to place coordinates along, in Ångström (`-length`) or pixels (`-length-px`). Default 500 Å. |
| `--twist` | Increment `rlnAngleRot` by this amount for each particle along a filament. |

### Examples

```
ais pick -d segmentations -t Ribosome -ou coordinates -threshold 128 -spacing 250 -size 1000000 -p 64
ais pick -d segmentations -t Microtubule -ou coordinates -filament -length 800 -spacing 82
```

When picking from Ais volumes, values are 0–255 and 128 is a good default threshold. Use the Ais 3D renderer or ChimeraX to find appropriate threshold, spacing, and size values that work for your target.
