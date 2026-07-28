"""
Ais training data (.scnt) I/O.

Two on-disk formats share the .scnt extension:

  * New format (preferred): an (uncompressed) tar archive holding a directory
    tree of per-box .mrc files - the same packaging idea as .scnm model files
    (see se_model.SEModel.save). Layout:

        metadata.json          apix, box_size, box_depth, features, n_samples,
                               input_flavours, annotated_flavour, format_version
        sources.json           { "<hash>": { aisTomogramName, ...coords... } }
        x_main/<hash>.mrc      input boxes for the annotated flavour, (D, H, W) float32
        x_<flavour>/<hash>.mrc  (optional) additional input flavours, same hashes
        y/<hash>.mrc           labels, (H, W) float32, values {0, 1, 2}

    Any directory whose name starts with "x_" is treated as an input flavour at
    training time, so end users can drop in their own flavours (e.g. x_wbp/)
    with .mrc files named by the same hashes as y/. The "annotated" flavour -
    the one the labels were drawn on - is recorded in metadata.json; it is the
    only flavour used for validation and is double-weighted when mixing.

  * Legacy format: a TIFF (renamed .scnt) holding a 4D array (N, D+1, H, W),
    D input slices followed by 1 label slice. Read-only support; never written.

The label encodes 0 = negative, 1 = positive, 2 = ignore (out-of-bounds / margin).
"""

import os
import glob
import json
import hashlib
import random
import shutil
import tarfile
import tempfile

import numpy as np
import mrcfile
import tifffile
from skimage.transform import resize
from Ais.core.normalization import global_stats, NORM_GLOBAL_MAD


FORMAT_VERSION = 1
INPUT_PREFIX = "x_"
LABEL_DIR = "y"
DEFAULT_ANNOTATED_FLAVOUR = "x_main"


# --------------------------------------------------------------------------- #
# box / label extraction (shared by the CLI and GUI extract code paths)
# --------------------------------------------------------------------------- #

def make_id(tomo_stem, feature_name, z, y, x):
    """Deterministic 16-hex-char id for one box. Flavour-independent, so every
    flavour of the same box shares a filename and links by name."""
    key = f"{tomo_stem}_{feature_name}_{z}_{y}_{x}"
    return hashlib.md5(key.encode()).hexdigest()[:16]


def _bin(arr, binning, anti_aliasing=True):
    if binning == 1:
        return arr
    j, k, l = arr.shape
    new_shape = [j, int(np.round(k / binning)), int(np.round(l / binning))]
    return resize(arr, new_shape, anti_aliasing=anti_aliasing, preserve_range=True,
                  order=1 if anti_aliasing else 0)


def extract_box(data, z, y, x, box_size, box_depth, n_slices):
    """Extract a (box_depth, box_size, box_size) box centred on (z, y, x) from a
    3D volume (Z, Y, X). Out-of-plane indices are clamped; in-plane the box is
    reflect-padded at the edges, and a validity mask marks the padded region."""
    box_size = int(box_size)
    box_depth = int(box_depth)
    z_indices = np.clip(np.arange(z - box_depth // 2, z + box_depth // 2 + 1), 0, n_slices - 1)
    zdim, ydim, xdim = data.shape

    depth = len(z_indices)
    half = box_size // 2
    y0, y1 = y - half, y + half
    x0, x1 = x - half, x + half

    box = np.zeros((depth, box_size, box_size), dtype=np.float32)
    validity = np.zeros((box_size, box_size), dtype=np.float32)

    iy0, iy1 = max(y0, 0), min(y1, ydim)
    ix0, ix1 = max(x0, 0), min(x1, xdim)

    if iy0 < iy1 and ix0 < ix1:
        pad_y = (iy0 - y0, y1 - iy1)
        pad_x = (ix0 - x0, x1 - ix1)
        for zi, zz in enumerate(z_indices):
            slc = data[zz, iy0:iy1, ix0:ix1]
            box[zi] = np.pad(slc, (pad_y, pad_x), mode='reflect')
        validity[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0] = 1

    return box, validity


def extract_label(feature, z, y, x, box_size):
    """Extract a (box_size, box_size) label patch from feature.slices[z]."""
    sl = feature.slices[z]
    h, w = sl.shape

    half = box_size // 2
    y0, y1 = y - half, y + half
    x0, x1 = x - half, x + half

    labels = np.zeros((box_size, box_size), dtype=sl.dtype)

    iy0, iy1 = max(y0, 0), min(y1, h)
    ix0, ix1 = max(x0, 0), min(x1, w)

    if iy0 < iy1 and ix0 < ix1:
        labels[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0] = sl[iy0:iy1, ix0:ix1]

    return labels


def extract_feature_samples(tomo_stem, flavour_data, annotated_flavour,
                            feature, box_size, box_depth, binning=1, is_negative=False,
                            tomo_path=None, on_box=None):
    """Extract all boxes for one feature of one annotated tomogram.

    flavour_data : dict {flavour_name -> 3D ndarray/memmap (Z, Y, X)}; must contain
                   annotated_flavour. Extra entries (e.g. x_iso) are additional flavours.
    is_negative  : if True, the label is all-background (used for negative examples).

    Returns a list of sample dicts: {hash, inputs:{flavour:(D,H,W)}, label:(H,W), source}.
    """
    box_size = int(box_size)
    box_depth = int(box_depth)
    samples = []
    box_coordinates = [(z, b[0], b[1]) for z in feature.boxes if feature.boxes[z] for b in feature.boxes[z]]
    annotation_box_size = int(getattr(feature, 'box_size', box_size))
    margin_per_side = max(0, (box_size - annotation_box_size) // 2)
    n_slices = flavour_data[annotated_flavour].shape[0]

    for z, x, y in box_coordinates:
        # validity comes from the annotated flavour (all flavours share dimensions)
        _, validity = extract_box(flavour_data[annotated_flavour], z, y, x, box_size, box_depth, n_slices)

        inputs = {}
        for flavour, data in flavour_data.items():
            if data.shape != flavour_data[annotated_flavour].shape:
                continue  # skip a flavour whose volume does not match the annotated one
            box, _ = extract_box(data, z, y, x, box_size, box_depth, n_slices)
            inputs[flavour] = _bin(box, binning)

        if is_negative or z not in feature.slices or feature.slices[z] is None:
            label = np.zeros((box_size, box_size), dtype=np.float32)
        else:
            label = extract_label(feature, z, y, x, box_size).astype(np.float32)

        label[validity == 0] = 2
        if margin_per_side > 0:
            label[:margin_per_side, :] = 2
            label[-margin_per_side:, :] = 2
            label[:, :margin_per_side] = 2
            label[:, -margin_per_side:] = 2
        label = _bin(label[None, :, :], binning, anti_aliasing=False)[0]

        h = make_id(tomo_stem, feature.title, z, y, x)
        samples.append({
            'hash': h,
            'inputs': inputs,
            'label': label,
            'source': {
                'aisTomogramName': tomo_stem,
                'aisTomogramPath': os.path.abspath(tomo_path) if tomo_path else '',
                'aisBoxCoordinateZ': int(z),
                'aisBoxCoordinateY': int(y),
                'aisBoxCoordinateX': int(x),
                'aisBoxSizeAnnotate': int(annotation_box_size),
                'aisBoxSizeExtracted': int(box_size),
                'aisFeatureName': feature.title,
                'aisNegative': bool(is_negative),
            },
        })
        if on_box is not None:
            on_box()

    return samples


# --------------------------------------------------------------------------- #
# writing
# --------------------------------------------------------------------------- #

def write_training_set(path, samples, apix, features,
                       annotated_flavour=DEFAULT_ANNOTATED_FLAVOUR, progress_cb=None):
    """Write a list of sample dicts (see extract_feature_samples) as a new-format
    .scnt tar archive. apix is the final pixel size of the extracted boxes (Angstrom).
    box_size / box_depth are derived from the data so metadata always matches the files."""
    if not samples:
        raise ValueError("write_training_set called with no samples.")
    input_flavours = sorted({f for s in samples for f in s['inputs']})
    sources = {s['hash']: s['source'] for s in samples}
    box_size = int(samples[0]['label'].shape[-1])
    box_depth = int(next(iter(samples[0]['inputs'].values())).shape[0])

    with tempfile.TemporaryDirectory() as tmp:
        for flavour in input_flavours:
            os.makedirs(os.path.join(tmp, flavour), exist_ok=True)
        os.makedirs(os.path.join(tmp, LABEL_DIR), exist_ok=True)

        n = len(samples)
        for i, s in enumerate(samples):
            for flavour, vol in s['inputs'].items():
                with mrcfile.new(os.path.join(tmp, flavour, s['hash'] + '.mrc'), overwrite=True) as m:
                    m.set_data(np.ascontiguousarray(vol, dtype=np.float32))
                    m.voxel_size = apix
            with mrcfile.new(os.path.join(tmp, LABEL_DIR, s['hash'] + '.mrc'), overwrite=True) as m:
                m.set_data(np.ascontiguousarray(s['label'], dtype=np.float32))
                m.voxel_size = apix
            if progress_cb is not None and n:
                progress_cb((i + 1) / n)

        metadata = {
            'format_version': FORMAT_VERSION,
            'apix': float(apix),
            'box_size': int(box_size),
            'box_depth': int(box_depth),
            'features': list(features),
            'n_samples': len(samples),
            'input_flavours': input_flavours,
            'annotated_flavour': annotated_flavour,
        }
        with open(os.path.join(tmp, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        with open(os.path.join(tmp, 'sources.json'), 'w') as f:
            json.dump(sources, f, indent=2)

        _tar_directory(tmp, path)


def _tar_directory(src_dir, out_path):
    """Pack the whole contents of src_dir into an (uncompressed) tar at out_path."""
    with tarfile.open(out_path, 'w') as tar:
        for root, _, files in os.walk(src_dir):
            for fn in files:
                full = os.path.join(root, fn)
                arc = os.path.relpath(full, src_dir).replace(os.sep, '/')
                tar.add(full, arcname=arc)


# --------------------------------------------------------------------------- #
# parallel extraction (scout-then-pool): one task per box, workers write their
# per-box .mrc files straight into a staging directory; the directory is tarred
# into a .scnt afterwards. The worker lives in this lightweight module so pool
# workers do not import tensorflow / glfw.
# --------------------------------------------------------------------------- #

_extract_worker_ctx = {}


def init_extract_worker(ctx):
    global _extract_worker_ctx
    _extract_worker_ctx = ctx


_stats_cache = {}   # path -> (center, scale)


def _display_view(data):
    # MRC mode-0 (int8) holds unsigned bytes in Ais; match SEFrame.get_slice
    return data.view(np.uint8) if data.dtype == np.int8 else data


def _volume_stats(path):
    if path not in _stats_cache:
        with mrcfile.mmap(path, mode='r', permissive=True) as m:
            _stats_cache[path] = global_stats(_display_view(m.data), method=NORM_GLOBAL_MAD)
    return _stats_cache[path]


def extract_box_task(task):
    """Worker: extract and write all flavours + label for one box.

    task keys: group, hash, flavour_paths{name:path}, annotated_flavour, z, y, x,
               box_size, box_depth, label_patch (ndarray or None), is_negative,
               margin_per_side, source.
    ctx keys (via init_extract_worker): staging_root, binning, apix.
    Returns (group, hash, source) on success, or None if the box was skipped.
    """
    ctx = _extract_worker_ctx
    staging_root = ctx['staging_root']
    binning = ctx['binning']
    apix = ctx['apix']

    box_size = int(task['box_size'])
    box_depth = int(task['box_depth'])
    z, y, x = task['z'], task['y'], task['x']
    group_dir = os.path.join(staging_root, task['group'])
    ann = task['annotated_flavour']

    validity = None
    wrote_any = False
    for flavour, path in task['flavour_paths'].items():
        if not os.path.exists(path):
            continue
        center, scale = _volume_stats(path)   # normalize by the parent volume
        with mrcfile.mmap(path, mode='r', permissive=True) as m:
            box, val = extract_box(_display_view(m.data), z, y, x, box_size, box_depth, m.data.shape[0])
        if flavour == ann:
            validity = val
        box = (box - center) / scale
        box = _bin(box, binning)
        out_dir = os.path.join(group_dir, flavour)
        os.makedirs(out_dir, exist_ok=True)
        with mrcfile.new(os.path.join(out_dir, task['hash'] + '.mrc'), overwrite=True) as m:
            m.set_data(np.ascontiguousarray(box, dtype=np.float32))
            m.voxel_size = apix
        wrote_any = True

    if validity is None or not wrote_any:
        return None  # annotated flavour missing - cannot build a usable sample

    if task['is_negative'] or task['label_patch'] is None:
        label = np.zeros((box_size, box_size), dtype=np.float32)
    else:
        label = np.array(task['label_patch'], dtype=np.float32)
    label[validity == 0] = 2
    m_per = task['margin_per_side']
    if m_per > 0:
        label[:m_per, :] = 2
        label[-m_per:, :] = 2
        label[:, :m_per] = 2
        label[:, -m_per:] = 2
    label = _bin(label[None, :, :], binning, anti_aliasing=False)[0]

    y_dir = os.path.join(group_dir, LABEL_DIR)
    os.makedirs(y_dir, exist_ok=True)
    with mrcfile.new(os.path.join(y_dir, task['hash'] + '.mrc'), overwrite=True) as m:
        m.set_data(np.ascontiguousarray(label, dtype=np.float32))
        m.voxel_size = apix

    return task['group'], task['hash'], task['source']


def pack_staging_dir(staging_dir, out_path, apix, features, sources,
                     annotated_flavour=DEFAULT_ANNOTATED_FLAVOUR):
    """Write metadata.json + sources.json into a worker-populated staging dir and
    tar it into a .scnt. box_size / box_depth are read back from the written files."""
    input_flavours = sorted(
        d for d in os.listdir(staging_dir)
        if d.startswith(INPUT_PREFIX) and os.path.isdir(os.path.join(staging_dir, d))
    )
    label_dir = os.path.join(staging_dir, LABEL_DIR)
    label_files = [f for f in os.listdir(label_dir) if f.endswith('.mrc')]
    n_samples = len(label_files)
    box_size = int(mrcfile.read(os.path.join(label_dir, label_files[0])).shape[-1])
    a_flavour = annotated_flavour if annotated_flavour in input_flavours else input_flavours[0]
    one_input = next(f for f in os.listdir(os.path.join(staging_dir, a_flavour)) if f.endswith('.mrc'))
    box_depth = int(mrcfile.read(os.path.join(staging_dir, a_flavour, one_input)).shape[0])

    metadata = {
        'format_version': FORMAT_VERSION,
        'apix': float(apix),
        'box_size': box_size,
        'box_depth': box_depth,
        'features': list(features),
        'n_samples': n_samples,
        'input_flavours': input_flavours,
        'annotated_flavour': a_flavour,
        'normalization': NORM_GLOBAL_MAD,
    }
    with open(os.path.join(staging_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    with open(os.path.join(staging_dir, 'sources.json'), 'w') as f:
        json.dump(sources, f, indent=2)

    _tar_directory(staging_dir, out_path)


# --------------------------------------------------------------------------- #
# reading
# --------------------------------------------------------------------------- #

def _normalize(arr):
    arr = arr.astype(np.float32)
    arr = arr - np.mean(arr)
    arr = arr / (np.std(arr) + 1e-7)
    return arr


def open_training_set(path):
    """Open a .scnt training set.

    `path` may be a single path (str) or a list/tuple of paths. Given several
    paths, their samples are pooled behind one training-set interface (see
    PooledTrainingSet); given one, the single set is returned directly.
    """
    if isinstance(path, (list, tuple)):
        paths = list(path)
        if len(paths) == 0:
            raise ValueError("open_training_set called with an empty list of paths.")
        if len(paths) == 1:
            return _open_single_training_set(paths[0])
        return PooledTrainingSet(paths)
    return _open_single_training_set(path)


def _open_single_training_set(path):
    """Open one .scnt file, auto-detecting the new (tar) or legacy (TIFF) format."""
    if tarfile.is_tarfile(path):
        return ScntTrainingSet(path)
    return LegacyTiffTrainingSet(path)


class ScntTrainingSet:
    """New-format (tar) training set. Boxes are read lazily per sample; an
    archive is extracted once to a managed temp directory for the lifetime of
    the object."""

    def __init__(self, path):
        self.path = path
        self._tmp = tempfile.mkdtemp(prefix='scnt_')
        with tarfile.open(path, 'r') as tar:
            tar.extractall(self._tmp)

        meta = {}
        meta_path = os.path.join(self._tmp, 'metadata.json')
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)

        self.input_flavours = sorted(
            d for d in os.listdir(self._tmp)
            if d.startswith(INPUT_PREFIX) and os.path.isdir(os.path.join(self._tmp, d))
        )
        if not self.input_flavours:
            raise ValueError(f"No input flavour directories (x_*) found in {path}")

        self.normalization = meta.get('normalization')   # None = legacy per-box; else global

        self.annotated_flavour = meta.get('annotated_flavour')
        if self.annotated_flavour not in self.input_flavours:
            if len(self.input_flavours) == 1:
                self.annotated_flavour = self.input_flavours[0]
            elif DEFAULT_ANNOTATED_FLAVOUR in self.input_flavours:
                self.annotated_flavour = DEFAULT_ANNOTATED_FLAVOUR
            else:
                self.annotated_flavour = self.input_flavours[0]
                print(f"Warning: no annotated_flavour recorded in {path}; "
                      f"using '{self.annotated_flavour}'.")

        label_dir = os.path.join(self._tmp, LABEL_DIR)
        self.hashes = sorted(os.path.splitext(f)[0] for f in os.listdir(label_dir) if f.endswith('.mrc'))
        self.n_samples = len(self.hashes)
        if self.n_samples == 0:
            raise ValueError(f"Training set {path} contains no samples.")

        sources_path = os.path.join(self._tmp, 'sources.json')
        if os.path.exists(sources_path):
            with open(sources_path) as f:
                self.sources = json.load(f)
        else:
            self.sources = {}

        # per-sample list of flavours that actually contain it (a flavour may cover
        # only a subset of samples), so mixing never reads a missing file
        self._flavours_for = {h: [] for h in self.hashes}
        for flavour in self.input_flavours:
            present = {os.path.splitext(n)[0]
                       for n in os.listdir(os.path.join(self._tmp, flavour))
                       if n.endswith('.mrc')}
            for h in self.hashes:
                if h in present:
                    self._flavours_for[h].append(flavour)

        first_in = self._read_input(self.hashes[0], self.annotated_flavour)
        self.box_depth = int(meta.get('box_depth', first_in.shape[0]))
        self.box_shape = int(meta.get('box_size', self._read_label(self.hashes[0]).shape[0]))
        self.apix = float(meta.get('apix', -1.0))
        if self.apix < 0:
            try:
                self.apix = float(mrcfile.open(self._input_path(self.hashes[0], self.annotated_flavour),
                                               header_only=True).voxel_size.x)
            except Exception:
                self.apix = -1.0

    def _input_path(self, h, flavour):
        return os.path.join(self._tmp, flavour, h + '.mrc')

    def _read_input(self, h, flavour):
        return np.array(mrcfile.read(self._input_path(h, flavour)), dtype=np.float32)

    def _read_label(self, h):
        return np.array(mrcfile.read(os.path.join(self._tmp, LABEL_DIR, h + '.mrc')), dtype=np.float32)

    def positive_indices(self):
        out = []
        for i, h in enumerate(self.hashes):
            if np.any(self._read_label(h) == 1):
                out.append(i)
        return out

    def get_sample(self, index, training=True):
        h = self.hashes[index]
        # only mix flavours present for this sample
        available = self._flavours_for.get(h) or [self.annotated_flavour]
        # legacy: normalize each box before mixing; global: boxes are pre-normalized at extract
        prep = (lambda a: a) if self.normalization == NORM_GLOBAL_MAD else _normalize
        if training and len(available) > 1:
            # 2/3 pure flavour (equal prob among available), 1/3 random-weight mixture;
            # flavours treated equally (no annotated double-weight).
            if random.random() < 2.0 / 3.0:
                vol = prep(self._read_input(h, random.choice(available)))
            else:
                fa, fb = random.sample(available, 2)
                a = prep(self._read_input(h, fa))
                b = prep(self._read_input(h, fb))
                f = random.uniform(0.0, 1.0)
                vol = a * f + b * (1.0 - f)
        else:
            flavour = self.annotated_flavour if self.annotated_flavour in available else available[0]
            vol = self._read_input(h, flavour)

        x = np.transpose(vol, (1, 2, 0)).astype(np.float32)        # (H, W, D)
        y = self._read_label(h)[:, :, None].astype(np.float32)      # (H, W, 1)
        return x, y

    def source_records(self):
        """Source dicts in sample (self.hashes) order; empty dict where missing."""
        return [self.sources.get(h, {}) for h in self.hashes]

    def flavour_coverage(self):
        """{flavour: number of samples that actually contain it}."""
        return {f: sum(1 for h in self.hashes if f in self._flavours_for.get(h, []))
                for f in self.input_flavours}

    def close(self):
        if self._tmp and os.path.isdir(self._tmp):
            shutil.rmtree(self._tmp, ignore_errors=True)
            self._tmp = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class PooledTrainingSet:
    """Pools the samples of several .scnt training sets behind the single
    training-set interface (n_samples, box_shape, box_depth, apix, input_flavours,
    positive_indices, get_sample, source_records, close).

    A global sample index maps to a (sub-set, local index) pair. Every pooled set
    must share box_size and box_depth, since those fix the model input shape; a
    differing pixel size is warned about but tolerated. Flavour mixing stays within
    each sub-set (see ScntTrainingSet.get_sample), so the pooled sets may carry
    different input flavours - the pooled input_flavours is just their union, for
    reporting."""

    def __init__(self, paths):
        self.path = list(paths)
        self.sets = [_open_single_training_set(p) for p in paths]

        box_shapes = {s.box_shape for s in self.sets}
        box_depths = {s.box_depth for s in self.sets}
        if len(box_shapes) > 1 or len(box_depths) > 1:
            dims = ', '.join(f"{os.path.basename(p)}: {s.box_shape}x{s.box_shape}x{s.box_depth}"
                             for s, p in zip(self.sets, paths))
            for s in self.sets:
                s.close()
            raise ValueError("Cannot pool training sets with different box dimensions "
                             f"({dims}). Re-extract them with matching --box-size and --box-depth.")
        self.box_shape = self.sets[0].box_shape
        self.box_depth = self.sets[0].box_depth

        apixes = [s.apix for s in self.sets]
        self.apix = apixes[0]
        if any(a > 0 and abs(a - self.apix) > 1e-3 for a in apixes):
            print(f"Warning: pooling training sets with differing pixel sizes {apixes}; "
                  f"using {self.apix} A/px for the model.")

        # global index -> (set index, local index), in set order then local order
        self._index = [(si, li) for si, s in enumerate(self.sets) for li in range(s.n_samples)]
        self.n_samples = len(self._index)

        flavours = set()
        for s in self.sets:
            flavours.update(s.input_flavours)
        self.input_flavours = sorted(flavours)

        # global only if every pooled set is global; else fall back to legacy per-box
        norms = {getattr(s, 'normalization', None) for s in self.sets}
        self.normalization = NORM_GLOBAL_MAD if norms == {NORM_GLOBAL_MAD} else None
        if len(norms) > 1:
            print("Warning: pooling training sets with mixed normalization schemes; "
                  "using per-box normalization for the model.")

    def positive_indices(self):
        out = []
        offset = 0
        for s in self.sets:
            out.extend(offset + li for li in s.positive_indices())
            offset += s.n_samples
        return out

    def get_sample(self, index, training=True):
        si, li = self._index[index]
        return self.sets[si].get_sample(li, training=training)

    def source_records(self):
        records = []
        for s in self.sets:
            records.extend(s.source_records())
        return records

    def flavour_coverage(self):
        out = {}
        for s in self.sets:
            if hasattr(s, "flavour_coverage"):
                for f, n in s.flavour_coverage().items():
                    out[f] = out.get(f, 0) + n
        return out

    def close(self):
        for s in self.sets:
            s.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class LegacyTiffTrainingSet:
    """Read-only legacy format: TIFF holding (N, D+1, H, W), D input slices then
    one label slice."""

    def __init__(self, path):
        self.path = path
        with tifffile.TiffFile(path) as tf:
            desc = tf.pages[0].description or ""
            try:
                self.apix = float(desc.split("apix=")[1].split()[0])
            except Exception:
                self.apix = -1.0
            data = tf.asarray()

        self._x = np.transpose(data[:, :-1, :, :], (0, 2, 3, 1))   # (N, H, W, D)
        self._y = data[:, -1, :, :, None]                          # (N, H, W, 1)
        self.n_samples, self.box_shape, _, self.box_depth = self._x.shape
        self.input_flavours = [DEFAULT_ANNOTATED_FLAVOUR]
        self.annotated_flavour = DEFAULT_ANNOTATED_FLAVOUR
        self.normalization = None   # legacy per-box
        self.sources = {}

    def positive_indices(self):
        return [i for i in range(self.n_samples) if np.any(self._y[i] == 1)]

    def get_sample(self, index, training=True):
        x = np.array(self._x[index], dtype=np.float32, copy=True)
        y = np.array(self._y[index], dtype=np.float32, copy=True)
        return x, y

    def source_records(self):
        return [{} for _ in range(self.n_samples)]

    def close(self):
        pass
