import os, sys, time, shutil, multiprocessing, glob, itertools, glfw, mrcfile, json, random
from collections import Counter
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # suppress TF C++ INFO and WARNING before import
from Ais.core.se_frame import SEFrame
from Ais.core.se_model import SEModel
from Ais.core.segmentation_editor import QueuedExport
from Ais.core.normalization import global_stats, NORM_GLOBAL_MAD
import Ais.core.config as cfg
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import numpy as np


def glfw_init():
    if not glfw.init():
        raise Exception("Could not initialize GLFW library for headless start!")
    glfw.window_hint(glfw.VISIBLE, False)
    window = glfw.create_window(1, 1, "invisible window", None, None)
    if not window:
        glfw.terminate()
        raise Exception("Could not create invisible window!")
    glfw.make_context_current(window)
    return window


def _pad_volume(volume):
    _, k, l = volume.shape
    pad_k = ((32 - (k % 32)) % 32) + 64
    pad_l = ((32 - (l % 32)) % 32) + 64

    padding_dim_k = (pad_k // 2, pad_k - pad_k // 2)
    padding_dim_l = (pad_l // 2, pad_l - pad_l // 2)
    volume = np.pad(volume, ((0, 0), padding_dim_k, padding_dim_l), mode='reflect')
    return volume, (*padding_dim_k, *padding_dim_l)


def _remove_padding(volume, padding):
    pt, pb, pl, pr = padding
    return volume[:, pt:None if pb == 0 else -pb, pl:None if pr == 0 else -pr]


def scale_volume_xy(volume, scale_factor):
    from skimage.transform import resize
    if np.abs(1.0 - scale_factor) < 0.05:
        return volume
    else:
        new_shape = np.round([volume.shape[0], volume.shape[1] * scale_factor, volume.shape[2] * scale_factor]).astype(int)
        return resize(volume, new_shape, anti_aliasing=True)


def _parse_input_for_slice(input_volume, j, model_depth, model_dimensionality):
    n_slices, _, _ = input_volume.shape
    if model_dimensionality == 2 or model_depth <= 1:
        return input_volume[np.clip(j, 0, n_slices - 1)][..., np.newaxis]
    half = model_depth // 2
    idx = np.clip(np.arange(j - half, j - half + model_depth), 0, n_slices - 1)
    slab = input_volume[idx, :, :]
    if model_dimensionality == 3:
        return np.transpose(slab, (1, 2, 0))[..., np.newaxis]
    else:
        return np.transpose(slab, (1, 2, 0))


def _infer_slab(model, vi, depth, z_start, z_end, batch_size, jitter_half=0):
    """Slab (3D) inference for a model that emits a full Z-slab: tile [z_start, z_end) into
    overlapping depth-`depth` slabs, run each, and blend the outputs in Z. Each slab slice is
    weighted by a trapezoid peaking over the trained reliable range [center +/- jitter_half]
    (uniform if jitter_half == 0). vi is (Z, Y, X); slices outside [z_start, z_end) are left at 0."""
    nz = vi.shape[0]
    stride = max(1, depth // 2)   # 50% Z overlap -> each slice covered by ~2 slabs
    if nz <= depth:
        starts = [0]
    else:
        hi_start = max(0, min(z_end, nz) - depth)
        starts = list(range(min(z_start, hi_start), hi_start + 1, stride))
        if starts[-1] != hi_start:
            starts.append(hi_start)

    # only the trained output positions [centre +/- jitter_half] contribute; positions outside that
    # range were never supervised (they saturate to garbage/all-ones), so weight them 0. Volume
    # top/bottom slices that only have untrained coverage then come out 0 (trims the all-ones margins).
    if jitter_half > 0:
        d = np.abs(np.arange(depth) - depth // 2)
        w = (d <= jitter_half).astype(np.float32)
    else:
        w = np.ones(depth, dtype=np.float32)

    acc = np.zeros_like(vi, dtype=np.float32)
    cnt = np.zeros(nz, dtype=np.float32)
    for bs in range(0, len(starts), batch_size):
        chunk = starts[bs:bs + batch_size]
        slabs = []
        for z0 in chunk:
            slab = vi[z0:z0 + depth]
            if slab.shape[0] < depth:   # volume thinner than one slab
                slab = np.pad(slab, ((0, depth - slab.shape[0]), (0, 0), (0, 0)), mode='reflect')
            slabs.append(np.transpose(slab, (1, 2, 0))[..., np.newaxis])   # (Y, X, depth, 1)
        out = np.squeeze(model(np.stack(slabs), training=False).numpy(), axis=-1)   # (B, Y, X, depth)
        for i, z0 in enumerate(chunk):
            z1 = min(z0 + depth, nz)
            d = z1 - z0
            acc[z0:z1] += w[:d, None, None] * np.transpose(out[i], (2, 0, 1))[:d]
            cnt[z0:z1] += w[:d]

    si = np.zeros_like(vi, dtype=np.float32)
    m = cnt > 0
    si[m] = acc[m] / cnt[m][:, None, None]
    si[:z_start] = 0.0
    si[z_end:] = 0.0
    return si


def _preprocess_tomo(tomo_path, model_apix, normalization=None):
    with mrcfile.open(tomo_path) as m:
        if normalization == NORM_GLOBAL_MAD and m.data.dtype == np.int8:
            volume = m.data.view(np.uint8).astype(np.float32)   # match GUI/extract int8 convention
        else:
            volume = m.data.astype(np.float32)
        volume_apix = float(m.voxel_size.x)
        in_voxel_size = m.voxel_size
        original_shape = volume.shape
    if model_apix is not None and volume_apix == 1.0:
        print(f'warning: {tomo_path} header lists voxel size as 1.0 A/px, which might be incorrect.')
    if volume_apix == 0.0:
        print(f'warning: volume apix is 0.0 so we cannot determine the scaling factor. we will assume the real pixel size is 10.0')
        volume_apix = 10.0
    global_norm = normalization == NORM_GLOBAL_MAD
    if global_norm:   # whole-volume stat at native resolution, before rescale
        center, scale = global_stats(volume)
        volume -= center
        volume /= scale
    if model_apix is not None:
        volume = scale_volume_xy(volume, volume_apix / float(model_apix))
    if not global_norm:   # legacy: per-slice
        for k in range(volume.shape[0]):
            sl = volume[k]
            sl -= sl.mean()
            sl /= sl.std() + 1e-6
            volume[k] = sl
    volume, padding = _pad_volume(volume)
    return {
        'volume': volume,
        'original_shape': original_shape,
        'padding': padding,
        'volume_apix': volume_apix,
        'in_voxel_size': in_voxel_size,
        'needs_resize': model_apix is not None,
    }


def _bin_volume_xy(vol, b=1):
    if b == 1:
        return vol
    else:
        j, k, l = vol.shape
        vol = vol[:, :k // b * b, :l // b * b]
        vol = vol.reshape((j, k // b, b, l // b, b)).mean(4).mean(2)
        return vol


def _segmentation_thread(model_path, data_paths, output_dir, gpu_id, test_time_augmentation=1, overwrite=False, model_apix=None, postprocessing_sigma=(0, 0, 0), batch_size=16, n_workers=4, center=100.0):
    from keras.models import clone_model
    from keras.layers import Input

    if isinstance(gpu_id, int):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    se_model = SEModel(no_glfw=True)
    se_model.load(model_path, compile=False)
    model_depth = se_model.model_depth
    model_dimensionality = 2.5 if model_depth > 1 else 2    # or it can be 3, see elif rank == 5 below
    model = se_model.model
    if model_apix is None:
        model_apix = se_model.apix
    input_shape = model.input_shape
    rank = len(input_shape)

    if rank == 4:
        new_input = Input(shape=(None, None, model_depth))
    elif rank == 5:
        model_dimensionality = 3
        new_input = Input(shape=(None, None, model_depth, 1))
    else:
        raise ValueError(f"Unsupported model input rank: {input_shape}")

    new_model = clone_model(model, input_tensors=new_input)
    new_model.set_weights(model.get_weights())
    # a slab model emits a full Z-slab (rank-5 output); it needs Z-tiled slab inference rather
    # than the per-slice window loop.
    is_slab = (len(new_model.output_shape) == 5)

    import threading
    import queue as _queue
    from concurrent.futures import ThreadPoolExecutor

    N = len(data_paths)
    start_time = time.time()
    completed = [0]
    processed = [0]
    last_completion_time = [start_time]

    def _eta_str():
        done = completed[0]
        remaining = N - done
        if processed[0] == 0:
            eta = "--:--:--"
        else:
            avg = (time.time() - start_time) / processed[0]
            remaining_secs = int(avg * remaining)
            eta = f"{remaining_secs // 3600:02d}:{(remaining_secs % 3600) // 60:02d}:{remaining_secs % 60:02d}"
        this_tomo_secs = int(time.time() - last_completion_time[0])
        this_tomo = f"{this_tomo_secs // 60:02d}:{this_tomo_secs % 60:02d}"
        return f"{eta} ({this_tomo} this tomo)"

    def _print(j, p, skipped=False):
        print(f"{j + 1}/{N} (GPU {gpu_id}) - {os.path.basename(model_path)} - {os.path.basename(p)}{' [skipped]' if skipped else ''} - eta {_eta_str()}")
        last_completion_time[0] = time.time()

    def _postprocess(j, p, out_path, seg, prepared):
        from scipy.ndimage import gaussian_filter, zoom
        try:
            if prepared['needs_resize']:
                sy = prepared['original_shape'][1] / seg.shape[1]
                sx = prepared['original_shape'][2] / seg.shape[2]
                seg = zoom(seg, (1.0, sy, sx), order=1, prefilter=False)
            if postprocessing_sigma != (0, 0, 0):
                seg = gaussian_filter(seg, np.array(postprocessing_sigma) / prepared['volume_apix'])
            seg = (seg * 255).astype(np.uint8)
            with mrcfile.new(out_path, overwrite=True) as mrc:
                mrc.set_data(seg)
                mrc.voxel_size = prepared['in_voxel_size']
        except Exception as e:
            print(f"Error postprocessing {p}:\n{e}")
            return
        completed[0] += 1
        processed[0] += 1
        _print(j, p)

    # shared pool handles both preproc and postproc — threads flow naturally toward whichever has work
    executor = ThreadPoolExecutor(max_workers=n_workers)

    # submitter thread: enqueues preproc futures; bounded queue limits memory
    preproc_q = _queue.Queue(maxsize=n_workers)

    def _submit_preproc():
        for j, p in enumerate(data_paths):
            out_path = os.path.join(output_dir, os.path.basename(os.path.splitext(p)[0]) + "__" + se_model.title + ".mrc")
            if os.path.exists(out_path) and not overwrite:
                preproc_q.put(('skip', j, p, out_path, None))
            else:
                preproc_q.put(('ready', j, p, out_path, executor.submit(_preprocess_tomo, p, model_apix, se_model.normalization)))
        preproc_q.put(None)

    threading.Thread(target=_submit_preproc, daemon=True).start()

    # --- inference loop (main thread, GPU) ---
    r = [0, 1, 2, 3, 0, 1, 2, 3]
    f = [0, 0, 0, 0, 1, 1, 1, 1]
    print(f"GPU {gpu_id} - starting inference with {model_dimensionality}D model '{se_model.title}' at {model_apix} A/px (n_workers = {n_workers}, tta = {test_time_augmentation}).")
    postproc_futures = []
    while True:
        item = preproc_q.get()
        if item is None:
            break
        kind, j, p, out_path, payload = item
        if kind == 'skip':
            completed[0] += 1
            _print(j, p, skipped=True)
            continue
        try:
            prepared = payload.result()  # wait for preproc to finish if not done yet
        except Exception as e:
            print(f"Error preprocessing {p}:\n{e}")
            continue
        try:
            with mrcfile.new(out_path, overwrite=True) as mrc:
                mrc.set_data(np.zeros((10, 10, 10), dtype=np.float32))
                mrc.voxel_size = 1.0
            volume = prepared['volume']
            padding = prepared['padding']
            pt, pb, pl, pr = padding
            seg = np.zeros((volume.shape[0], volume.shape[1] - pt - pb, volume.shape[2] - pl - pr), dtype=np.float32)
            for k in range(test_time_augmentation):
                vi = np.rot90(volume, k=r[k], axes=(1, 2))
                if f[k]: vi = np.flip(vi, axis=2)
                si = np.zeros_like(vi, dtype=np.float32)
                n_z = vi.shape[0]
                frac = np.clip(center / 100.0, 0.0, 1.0)
                n_center = max(1, int(round(n_z * frac)))
                z_start = (n_z - n_center) // 2
                z_end = z_start + n_center
                if is_slab:
                    si = _infer_slab(new_model, vi, model_depth, z_start, z_end, batch_size, se_model.z_jitter // 2)
                else:
                    for bs in range(z_start, z_end, batch_size):
                        bjs = list(range(bs, min(bs + batch_size, z_end)))
                        inp = np.stack([_parse_input_for_slice(vi, bj, model_depth, model_dimensionality) for bj in bjs])
                        res = new_model(inp, training=False).numpy()
                        for i, bj in enumerate(bjs):
                            si[bj] = np.squeeze(res[i])
                if f[k]: si = np.flip(si, axis=2)
                si = np.rot90(si, k=-r[k], axes=(1, 2))
                seg += _remove_padding(si, padding)
            seg = np.clip(seg / test_time_augmentation, 0.0, 1.0)
            postproc_futures.append(executor.submit(_postprocess, j, p, out_path, seg, prepared))
        except Exception as e:
            print(f"Error segmenting {p}:\n{e}")

    for fut in postproc_futures:
        fut.result()
    executor.shutdown(wait=False)


def dispatch_parallel_segment(model_path, data_patterns, output_directory, gpus, test_time_augmentation=1, parallel=1, overwrite=0, processing_apix=None, postprocessing_sigma=(0, 0, 0), batch_size=16, n_workers=None, center=100.0):
    if n_workers is None:
        n_workers = min(16, max(1, (os.cpu_count() or 1) // len(gpus)))
    if not os.path.isabs(model_path):
        model_path = os.path.join(os.getcwd(), model_path)

    # Normalise data_directory to a list of patterns/paths
    if isinstance(data_patterns, (list, tuple)):
        patterns = list(data_patterns)
    else:
        patterns = [data_patterns]

    # Make patterns absolute where appropriate
    abs_patterns = []
    for p in patterns:
        if not os.path.isabs(p):
            p = os.path.join(os.getcwd(), p)
        abs_patterns.append(p)
    patterns = abs_patterns

    # Make output directory absolute
    if not os.path.isabs(output_directory):
        output_directory = os.path.join(os.getcwd(), output_directory)
    os.makedirs(output_directory, exist_ok=True)

    # Collect all .mrc inputs from all patterns
    all_data_paths = []
    for p in patterns:
        if p.endswith('.txt') and os.path.isfile(p):
            # .txt file with one tomogram path per line (e.g. a pom subset) (260331)
            with open(p) as f:
                for line in f:
                    entry = line.strip()
                    if entry and not entry.endswith('.mrc'):
                        entry += '.mrc'
                    if entry:
                        all_data_paths.append(entry)
        elif os.path.isdir(p):
            # Treat as directory: pick up all .mrc files
            matches = glob.glob(os.path.join(p, "*.mrc"))
            all_data_paths.extend(matches)
        else:
            # Treat as glob pattern or explicit file
            matches = glob.glob(p)
            all_data_paths.extend(matches)

    # dedupe + sort to a canonical order, then shuffle with a fixed seed: a many-dataset run
    # interleaves datasets (a representative mix sooner) but the order is reproducible - the same
    # set of tomograms always yields the same (pseudo-random) order
    all_data_paths = sorted(f for f in set(all_data_paths) if os.path.splitext(f)[-1] == ".mrc")
    random.Random(0).shuffle(all_data_paths)

    if len(all_data_paths) == 0:
        print(f"No .mrc files found for data_directory={patterns}. Nothing to do.")
        return

    # Divide work over GPUs
    data_div = {gpu: [] for gpu in gpus}
    for gpu, data_path in zip(itertools.cycle(gpus), all_data_paths):
        data_div[gpu].append(data_path)

    if parallel == 1:
        # One process per GPU
        processes = []
        for gpu_id in data_div:
            p = multiprocessing.Process(
                target=_segmentation_thread,
                args=(
                    model_path,
                    data_div[gpu_id],
                    output_directory,
                    gpu_id,
                    test_time_augmentation,
                    overwrite,
                    processing_apix,
                    postprocessing_sigma,
                    batch_size,
                    n_workers,
                    center
                ),
            )
            processes.append(p)
            p.start()

        for p in processes:
            p.join()
    else:
        # Single process using all GPUs (gpu_id is a comma-separated list)
        gpu_id_str = ",".join(str(n) for n in gpus)
        _segmentation_thread(
            model_path,
            all_data_paths,
            output_directory,
            gpu_id_str,
            test_time_augmentation,
            overwrite,
            processing_apix,
            postprocessing_sigma,
            batch_size,
            n_workers,
            center
        )


def print_available_model_architectures():
    model = SEModel(no_glfw=True)
    model.load_models()

    first_col = [f"index: {j} (-a {j})" for j in range(len(SEModel.AVAILABLE_MODELS))]
    col_width = max(len(text) for text in first_col) + 2

    for j, key in enumerate(SEModel.AVAILABLE_MODELS):
        print(f"{first_col[j]:<{col_width}}architecture name: {key}")


def resolve_model_architecture(architecture):
    # Resolve the -a / --model_architecture argument (an integer index OR a model title) to the
    # integer index into SEModel.AVAILABLE_MODELS that the rest of the code expects.
    # Returns None for None (keeps the default-architecture behaviour). Exits non-zero if it can't resolve.
    if architecture is None:
        return None
    if not SEModel.MODELS_LOADED:
        SEModel.load_models()
    n = len(SEModel.AVAILABLE_MODELS)

    # an integer index (either an int or an all-digit string)
    if isinstance(architecture, int) or (isinstance(architecture, str) and architecture.strip().lstrip('-').isdigit()):
        index = int(architecture)
        if 0 <= index < n:
            return index
        print(f"Error: model architecture index {index} is out of range (valid indices: 0 - {n - 1}).")
        print_available_model_architectures()
        exit(1)

    # a model title; match case-insensitively, treating '-', '_' and whitespace as equivalent
    def normalize(text):
        return " ".join(str(text).replace("_", " ").replace("-", " ").split()).lower()

    target = normalize(architecture)
    for j, title in enumerate(SEModel.AVAILABLE_MODELS):
        if normalize(title) == target:
            return j

    print(f"Error: could not find a model architecture matching '{architecture}'.")
    print_available_model_architectures()
    exit(1)


def train_model(training_data, output_directory, architecture=None, epochs=50, batch_size=32, negatives=0.0, copies=4, model_path='', gpus="0", parallel=1, rate=1e-3, name="Unnamed model", extra_augmentations=False):
    import keras.callbacks

    architecture = resolve_model_architecture(architecture)

    class CheckpointCallback(keras.callbacks.Callback):
        def __init__(self, se_model, path):
            super().__init__()
            self.se_model = se_model
            self.path = path
            self.best_loss = 1e9

        def on_epoch_end(self, epoch, logs=None):
            loss = logs.get('loss', None)
            if loss is not None and loss < self.best_loss:
                self.best_loss = loss
                self.se_model.save(self.path)

    # training_data may be a single path or several; multiple .scnt files are pooled during training.
    if isinstance(training_data, (list, tuple)):
        training_data_paths = list(training_data)
    else:
        training_data_paths = [training_data]
    training_data_paths = [p if os.path.isabs(p) else os.path.join(os.getcwd(), p) for p in training_data_paths]
    if not os.path.isabs(output_directory):
        output_directory = os.path.join(os.getcwd(), output_directory)
    os.makedirs(output_directory, exist_ok=True)
    if model_path and not os.path.isabs(model_path):
        model_path = os.path.join(os.getcwd(), model_path)
    if len(training_data_paths) == 1:
        print(f"training data: {training_data_paths[0]}")
    else:
        print(f"training data ({len(training_data_paths)} files, samples pooled):")
        for p in training_data_paths:
            print(f"  {p}")
    print(f"output directory: {output_directory}")
    model = SEModel(no_glfw=True)
    model.load_models()
    model.title = name
    if model_path:
        # continue from a saved model. load() overwrites title too, so re-apply -name if given.
        print(f"continuing training from {model_path}")
        model.load(model_path)
        if name != "Unnamed model":
            model.title = name
        print(f"  architecture: {SEModel.AVAILABLE_MODELS[model.model_enum]}, box {model.box_size}-{model.model_depth}, {model.apix:.1f} A/px")
    elif architecture is None:
        model.model_enum = SEModel.DEFAULT_MODEL_ENUM
        print(f"using default model architecture: {SEModel.AVAILABLE_MODELS[SEModel.DEFAULT_MODEL_ENUM]}")
    else:
        model.model_enum = architecture
        print(f"using model architecture {architecture}: {SEModel.AVAILABLE_MODELS[architecture]}")

    model.train_data_path = training_data_paths if len(training_data_paths) > 1 else training_data_paths[0]
    model.epochs = epochs
    model.batch_size = batch_size
    model.excess_negative = int((100 * negatives) - 100)
    model.n_copies = copies

    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    checkpoint_callback = CheckpointCallback(model, os.path.join(output_directory, f"{model.title}{cfg.filetype_semodel}"))
    # train() spawns a worker thread; the strategy scope must be entered inside that thread
    # (in _train, around build/compile/fit), not here. Wrapping the thread-spawning train()
    # call did nothing for the off-thread fit, and on the -m path clear_session() ran inside
    # the scope and emptied TF's strategy stack -> IndexError on scope exit.
    strategy = tf.distribute.MirroredStrategy() if parallel else None
    model.train(rate=rate, external_callbacks=[checkpoint_callback], extra_augmentations=extra_augmentations, strategy=strategy)

    while model.background_process_train.progress < 1.0:
        time.sleep(0.2)

    time.sleep(10.0)
    file_path = os.path.join(output_directory, f"{model.title}{cfg.filetype_semodel}")
    model.load(os.path.join(output_directory, f"{model.title}{cfg.filetype_semodel}"))
    model.toggle_inference()
    model.save(file_path)
    print(f"\nDone training {os.path.join(output_directory, f'{model.title}{cfg.filetype_semodel}')}")


def _pick_tomo(tomo_path, output_path, margin, threshold, binning, spacing, size, spacing_px, size_px, verbose, filament=False, filament_length=500.0, filament_length_px=None, centroid=False, min_particles=0, twist_per_sample=0.0, orient=None, orient_sign='z'):
    # find right values for spacing and size.
    voxel_size = mrcfile.open(tomo_path, permissive=True, header_only=True).voxel_size.x
    if voxel_size == 0.0:
        print(f"warning: {tomo_path} has voxel size 0.0")
        voxel_size = 10.0
    if spacing_px is None:
        min_spacing = spacing / 10.0
    else:
        min_spacing = spacing_px * voxel_size / 10.0

    if size_px is None:
        min_size = size / 1000.0
    else:
        min_size = size_px * (voxel_size / 10.0)**3

    if filament_length_px is None:
        filament_length = filament_length / 10.0
    else:
        filament_length = filament_length_px * (voxel_size / 10.0)

    if filament:
        from Ais.core.filaments import pick_filament
        return pick_filament(mrcpath=tomo_path, out_path=output_path, margin=margin, threshold=threshold, binning=binning, spacing_nm=min_spacing, size_nm=min_size, pixel_size=voxel_size / 10.0, min_length=filament_length, twist_per_sample=twist_per_sample)
    else:
        from Ais.core.util import pick_particles
        return pick_particles(mrcpath=tomo_path, out_path=output_path, margin=margin, threshold=threshold, binning=binning, min_spacing=min_spacing, min_size=min_size, pixel_size=voxel_size / 10.0, verbose=verbose, centroid=centroid, min_particles=min_particles, orient=orient, orient_sign=orient_sign)


def _clr_print(txt, clr):
    colors = {
        "none": "\033[37m",
        "few": "\033[33m",
        "mid": "\033[36m",
        "many": "\033[32m",
        "red": "\033[31m"
    }
    print(f"{colors[clr]}{txt}\033[0m")


def _picking_thread(data_paths, output_directory, margin, threshold, binning, spacing, size, spacing_px, size_px, process_id, verbose, filament=False, filament_length=500.0, filament_length_px=None, centroid=False, min_particles=0, twist_per_sample=0.0, orient=None, orient_sign='z'):
    try:
        for j, p in enumerate(data_paths):
            out_path = os.path.join(output_directory, os.path.splitext(os.path.basename(p))[0]+"_coords.star")
            n_particles, n_filaments = _pick_tomo(p, out_path, margin, threshold, binning, spacing, size, spacing_px, size_px, verbose, filament, filament_length, filament_length_px, centroid, min_particles, twist_per_sample, orient, orient_sign)

            if n_particles < min_particles:
                _clr_print(
                    f"{j + 1}/{len(data_paths)} (process {process_id}) - {n_particles} {'particles' if not filament else f'coordinates in {n_filaments} filaments'} in {os.path.basename(p)}", 'red')
                continue

            clr = 'none'
            if 0 < n_particles < 10:
                clr = 'few'
            elif 10 <= n_particles < 50:
                clr = 'mid'
            elif n_particles >= 50:
                clr = 'many'
            _clr_print(f"{j+1}/{len(data_paths)} (process {process_id}) - {n_particles} {'particles' if not filament else f'coordinates in {n_filaments} filaments'} in {os.path.basename(p)}", clr)
    except KeyboardInterrupt:
        pass


def _read_subset_txt(subset_path):
    """Read a Pom-style subset .txt file and return a set of bare tomogram names."""
    names = set()
    with open(subset_path) as f:
        for line in f:
            entry = line.strip()
            if not entry:
                continue
            base = entry.replace('\\', '/').rsplit('/', 1)[-1]
            if base.endswith('.mrc'):
                base = base[:-4]
            names.add(base)
    return names


def dispatch_parallel_pick(target, data_directory, output_directory, margin, threshold, binning, spacing, size, parallel=1, spacing_px=None, size_px=None, verbose=False, pom_capp_config="", filament=False, filament_length=500.0, centroid=False, min_particles=0, twist_per_sample=0.0, subset=None, orient=None, orient_sign='z'):
    data_directory = os.path.abspath(data_directory)
    output_directory = os.path.abspath(output_directory)

    os.makedirs(output_directory, exist_ok=True)
    all_data_paths = glob.glob(os.path.join(data_directory, f"*__{target}.mrc"))

    if subset is not None:
        subset_names = _read_subset_txt(subset)
        n_before = len(all_data_paths)
        all_data_paths = [p for p in all_data_paths if os.path.basename(p).split(f'__{target}')[0] in subset_names]
        print(f'Subset {os.path.basename(subset)}: {len(all_data_paths)}/{n_before} segmented volumes matched.')

    pom_capp_info_str = ""
    if pom_capp_config:
        with open(pom_capp_config, 'r') as f:
            config = json.load(f)
            subsets = config.get('subsets', ['all'])
            subsets = ['all'] if not subsets else subsets
            if not 'all' in subsets:
                all_data_paths = []
                for s in subsets:
                    subset_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(pom_capp_config))), 'subsets', f'{s}.json')
                    with open(subset_path, 'r') as sf:
                        subset = json.load(sf)['tomos']
                        print(subset)
                        for t in subset:
                            all_data_paths.append(os.path.join(data_directory, f"{t}__{target}.mrc"))


        pom_capp_info_str = f"in selected Pom subsets: {', '.join(subsets)}"
    print(f'Found {len(all_data_paths)} files with pattern {os.path.join(data_directory, f"*__{target}.mrc")} {pom_capp_info_str}. Picking in {"blob" if not filament else "filament"} mode.')
    data_div = {p_id: list() for p_id in range(parallel)}
    for p_id, data_path in zip(itertools.cycle(range(parallel)), all_data_paths):
        data_div[p_id].append(data_path)

    processes = []
    try:
        for p_id in data_div:
            p = multiprocessing.Process(target=_picking_thread,
                                        args=(data_div[p_id], output_directory, margin, threshold, binning, spacing, size, spacing_px, size_px, p_id, verbose, filament, filament_length, None, centroid, min_particles, twist_per_sample, orient, orient_sign))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()
    except KeyboardInterrupt:
        for p in processes:
            if p.is_alive():
                p.terminate()
                p.join(timeout=1)


def _find_flavours(tomo_path):
    """this method is specific to the easymode development environment and shouldn't be called by end users"""
    flavours = {}
    stem = os.path.splitext(os.path.basename(tomo_path))[0]
    # Annotated (cryocare) volumes are kept flat in <easymode>/volumes_cryocare/, so
    # the easymode root is two levels up from the annotated tomogram.
    base = os.path.dirname(os.path.dirname(tomo_path))   # .../easymode

    # isonet: flat alongside the cryocare volumes
    iso = os.path.join(base, 'volumes_isonet', stem + '.mrc')
    if os.path.exists(iso):
        flavours['x_iso'] = iso

    return flavours


class _ScanDisplay:
    """Live multi-line console region for `ais extract` scanning: a tqdm-style progress bar over
    the annotated tomograms plus one in-place 'boxes found' line per feature, rewritten with ANSI
    cursor moves (this module already uses ANSI colour). The bar string is produced by tqdm's own
    format_meter() so it matches the extraction bar exactly (same glyphs, %, it/s, elapsed<eta).
    Non-TTY: prints one summary at the end instead of animating."""

    def __init__(self, total, features, annotated_flavour):
        self.total = max(1, total)
        self.features = list(features)
        self.main = annotated_flavour
        self.namew = max((len(f) for f in self.features), default=0)
        self.n = 0
        self.box = {f: 0 for f in self.features}
        self.tomos = {f: 0 for f in self.features}
        self.flav = {f: Counter() for f in self.features}
        self.tty = sys.stdout.isatty()
        self.nlines = 1 + len(self.features)
        self.t0 = time.time()
        self._drawn = False
        # format_meter() defaults to Unicode block glyphs; fall back to ASCII when the stream
        # can't encode them (matches how the tqdm object behaves on a non-UTF console).
        try:
            from tqdm.utils import _supports_unicode
            self.ascii = not _supports_unicode(sys.stdout)
        except Exception:
            self.ascii = 'utf' not in ((getattr(sys.stdout, 'encoding', '') or '').lower())

    @staticmethod
    def _ncols():
        return max(20, shutil.get_terminal_size(fallback=(100, 24)).columns - 1)

    def _bar_line(self, ncols):
        from tqdm import tqdm as _tqdm
        return _tqdm.format_meter(self.n, self.total, time.time() - self.t0, ncols=ncols,
                                  prefix='scanning tomograms', unit='tomo', ascii=self.ascii)

    def _feat_line(self, f, ncols):
        unit = 'tomogram' if self.tomos[f] == 1 else 'tomograms'
        txt = f"{f + ':':<{self.namew + 1}} {self.box[f]:>7} boxes found in {self.tomos[f]} {unit}"
        extras = sorted(k for k in self.flav[f] if k != self.main)
        if extras:
            parts = [f"{self.main} {self.flav[f][self.main]}"] + [f"{k} {self.flav[f][k]}" for k in extras]
            txt += f"  (flavours: {', '.join(parts)})"
        if len(txt) > ncols:
            ell = '...' if self.ascii else '…'
            txt = txt[:max(0, ncols - len(ell))] + ell
        return f"\033[96m{txt}\033[0m"

    def _render(self):
        ncols = self._ncols()
        out = f"\033[{self.nlines}A" if self._drawn else ""
        for ln in [self._bar_line(ncols)] + [self._feat_line(f, ncols) for f in self.features]:
            out += "\033[2K" + ln + "\n"
        sys.stdout.write(out); sys.stdout.flush()
        self._drawn = True

    def start_tomo(self):
        self.n += 1
        if self.tty:
            self._render()

    def add(self, feature, nboxes, flavours):
        self.box[feature] += nboxes
        self.tomos[feature] += 1
        for fl in flavours:
            self.flav[feature][fl] += nboxes

    def finish(self):
        if self.tty:
            self._render()
        else:
            ncols = self._ncols()
            print(self._bar_line(ncols))
            for f in self.features:
                print(self._feat_line(f, ncols))


def extract_training_data(features, data_directory, output_directory, box_size, box_depth, exclude=None, merge=False, coordinates=False, apix=10.0, easymode=False):
    import pickle, tempfile, shutil
    import starfile, pandas as pd
    import Ais.core.se_scnt as se_scnt
    from tqdm import tqdm
    from collections import Counter

    MERGED_GROUP = "__merged__"
    # for any slab (box_depth > 1) store 4 extra slices top & bottom, so 3D training can jitter the
    # annotated slice's Z-position within the box (breaks the "only the centre slice is learned"
    # failure mode). box_depth stays the MODEL depth; the .scnt just carries the extra context.
    Z_JITTER = 8 if box_depth > 1 else 0
    stored_depth = box_depth + Z_JITTER
    annotated_tomograms = glob.glob(os.path.join(data_directory, "*.scns"))

    excluded_files = []
    if exclude is not None:
        for e in exclude:
            if e.endswith('.txt'):
                with open(e) as f:
                    excluded_files.extend([line.strip() for line in f if line.strip()])
            elif '*' in e:
                excluded_files.extend(glob.glob(e))
            else:
                excluded_files.append(e)
    excluded_files = [os.path.basename(os.path.splitext(f)[0]) for f in excluded_files]

    print(f'scanning {len(annotated_tomograms)} annotated tomograms for {len(features)} features...')

    coord_rows = {f: [] for f in features} if coordinates else None
    tasks = []                                   # one task per box (for parallel extraction)
    feature_box_count = {f: 0 for f in features}
    notes = []                                   # warnings/info, emitted below the live region
    disp = _ScanDisplay(len(annotated_tomograms), features, se_scnt.DEFAULT_ANNOTATED_FLAVOUR)
    apix = float(apix)   # target pixel size for the extracted boxes (written to header + filename)

    # ---- scout: load every .scns, collect box coordinates + label patches ----
    for annotation in annotated_tomograms:
        disp.start_tomo()
        stem = os.path.splitext(os.path.basename(annotation))[0]
        if stem in excluded_files:
            notes.append('\033[38;5;208m' + f'{os.path.basename(annotation)} - excluded' + '\033[0m')
            continue

        try:
            with open(annotation, 'rb') as pf:
                se_frame = pickle.load(pf)
        except Exception as e:
            notes.append(f"error loading {os.path.basename(annotation)}: {e}")
            continue

        tomo = se_frame.path
        if not os.path.exists(tomo):
            tomo = os.path.join(os.path.dirname(annotation), os.path.basename(se_frame.path.replace('\\','/')))

        tomo_stem = os.path.splitext(os.path.basename(tomo))[0]
        if tomo_stem in excluded_files:
            notes.append('\033[38;5;208m' + f'{os.path.basename(annotation)} - excluded' + '\033[0m')
            continue

        if not coordinates and not os.path.exists(tomo):
            notes.append('\033[38;5;208m' + f'tomogram not found at {tomo} - skipping' + '\033[0m')
            continue

        tomo_mrc_name = os.path.basename(tomo).split("__")[0] + ".mrc"

        # native pixel size from this tomogram's header; each box is extracted at native scale then
        # resampled in XY to the target --apix (Z preserved, matching inference), so mixed-apix
        # tomograms land on a common grid. When native ~= target (<5%) the box is kept as-is.
        native_bs = int(box_size)
        if not coordinates:
            native_apix = float(mrcfile.open(tomo, header_only=True).voxel_size.x)
            if native_apix <= 0 or abs(native_apix - 1.0) < 1e-6:
                notes.append('\033[38;5;208m' + f'{os.path.basename(tomo)}: header pixel size {native_apix} A/px looks unset; treating as --apix {apix:.2f} (no rescale)' + '\033[0m')
                native_apix = apix
            if abs(native_apix / apix - 1.0) >= 0.05:
                native_bs = int(round(box_size * apix / native_apix))

        if coordinates:
            for f in se_frame.features:
                if f.title not in features:
                    continue
                box_coordinates = [(z, box[0], box[1]) for z in f.boxes for box in f.boxes[z]]
                disp.add(f.title, len(box_coordinates), [])
                for z, k, l in box_coordinates:
                    coord_rows[f.title].append({
                        'rlnCoordinateZ': z,
                        'rlnCoordinateY': l,
                        'rlnCoordinateX': k,
                        'rlnMicrographName': tomo_mrc_name,
                    })
            continue

        # flavours are found lazily: only once this tomogram actually contributes boxes
        # for a requested feature (many .scns won't have the feature of interest).
        flavour_paths = None

        for f in se_frame.features:
            if f.title not in features:
                continue
            ann_bs = getattr(f, 'box_size', native_bs)
            margin_per_side = max(0, (native_bs - ann_bs) // 2)   # native px; scaled with the label
            box_coordinates = [(z, b[0], b[1]) for z in f.boxes if f.boxes[z] for b in f.boxes[z]]
            if not box_coordinates:
                continue
            if flavour_paths is None:
                # x_main is the annotated (cryocare) flavour; --easymode adds any others found
                flavour_paths = {se_scnt.DEFAULT_ANNOTATED_FLAVOUR: tomo}
                if easymode:
                    flavour_paths.update(_find_flavours(tomo))
            disp.add(f.title, len(box_coordinates), flavour_paths.keys())
            group = MERGED_GROUP if merge else f.title
            for z, x, y in box_coordinates:
                if z in f.slices and f.slices[z] is not None:
                    label_patch = se_scnt.extract_label(f, z, y, x, native_bs)
                else:
                    label_patch = None
                tasks.append({
                    'group': group,
                    'hash': se_scnt.make_id(tomo_stem, f.title, z, y, x),
                    'flavour_paths': dict(flavour_paths),
                    'annotated_flavour': se_scnt.DEFAULT_ANNOTATED_FLAVOUR,
                    'z': int(z), 'y': int(y), 'x': int(x),
                    'box_size': int(native_bs), 'out_box_size': int(box_size),
                    'box_depth': int(stored_depth),
                    'label_patch': label_patch,
                    'is_negative': False,
                    'margin_per_side': margin_per_side,
                    'source': {
                        'aisTomogramName': tomo_stem,
                        'aisTomogramPath': os.path.abspath(tomo),
                        'aisBoxCoordinateZ': int(z),
                        'aisBoxCoordinateY': int(y),
                        'aisBoxCoordinateX': int(x),
                        'aisBoxSizeAnnotate': int(ann_bs),
                        'aisBoxSizeExtracted': int(box_size),
                        'aisFeatureName': f.title,
                    },
                })
                feature_box_count[f.title] += 1

    disp.finish()
    for note in notes:
        print(note)
    print()

    os.makedirs(output_directory, exist_ok=True)

    if coordinates:
        if merge:
            all_rows = []
            for f in features:
                for row in coord_rows[f]:
                    row['aisFeature'] = f
                    all_rows.append(row)
            df = pd.DataFrame(all_rows)
            merged_name = "_".join(features)
            out_path = os.path.join(output_directory, f'{merged_name}_coordinates.star')
            starfile.write({'particles': df}, out_path, overwrite=True)
            print(f'Wrote {len(df)} coordinates to {out_path}')
        else:
            for f in features:
                if not coord_rows[f]:
                    print(f'\033[96m{f}: 0 coordinates. Skipping.\033[0m')
                    continue
                df = pd.DataFrame(coord_rows[f])
                out_path = os.path.join(output_directory, f'{f}_coordinates.star')
                starfile.write({'particles': df}, out_path, overwrite=True)
                print(f'Wrote {len(df)} {f} coordinates to {out_path}')
        return

    if not tasks:
        print('\033[96mNo training boxes for any feature. Skipping export.\033[0m')
        return

    apix_final = apix   # boxes are resampled to the target apix, so the .scnt is at this scale
    _apix_tag = f"_{apix_final:.2f}Apx"

    # ---- dispatch: extract boxes one feature at a time, saving each feature's .scnt as soon
    # as it finishes. A single SHARED pool is reused across features (not one pool per feature)
    # so the per-feature bar + incremental save don't pay the worker-spawn cost K times. ----
    n_proc = max(1, min(16, os.cpu_count() or 1, len(tasks)))
    staging_root = tempfile.mkdtemp(prefix='scnt_extract_')
    ctx = {'staging_root': staging_root, 'apix': apix_final}

    # per-box flavour sets, for the "(x_main N, additional flavours ...)" report
    hash_flavours = {t['hash']: tuple(t['flavour_paths'].keys()) for t in tasks}
    _main = se_scnt.DEFAULT_ANNOTATED_FLAVOUR

    def _flavour_detail(sources_dict):
        c = Counter()
        for h in sources_dict:
            for flav in hash_flavours.get(h, ()):
                c[flav] += 1
        extras = sorted(k for k in c if k != _main)
        if not extras:
            return ""
        return f" ({_main} {c.get(_main, 0)}, additional flavours {', '.join(f'{k} ({c[k]})' for k in extras)})"

    # ordered work units: (label, staging-subdir/group key, .scnt name stem, feature list, tasks)
    if merge:
        names = [f for f in features if feature_box_count[f] > 0]
        groups = [('Merged', MERGED_GROUP, "_".join(names), names, tasks)]
    else:
        groups = [(f, f, f, [f], [t for t in tasks if t['group'] == f]) for f in features]

    try:
        print('\033[96m' + f'extracting {len(tasks)} boxes using {n_proc} process(es)...' + '\033[0m')
        pool = None if n_proc == 1 else multiprocessing.Pool(
            processes=n_proc, initializer=se_scnt.init_extract_worker, initargs=(ctx,))
        if pool is None:
            se_scnt.init_extract_worker(ctx)
        try:
            for label, group_key, name_stem, feats_list, gtasks in groups:
                out_path = f"{box_size}x{box_size}x{box_depth}{_apix_tag}_{name_stem}.scnt"
                if not gtasks:
                    print('\033[96m' + f'{label}: 0 training boxes. Skipping export.' + '\033[0m')
                    continue
                if pool is None:
                    stream = (se_scnt.extract_box_task(t) for t in gtasks)
                else:
                    stream = pool.imap_unordered(se_scnt.extract_box_task, gtasks, chunksize=8)
                sources = {}
                for r in tqdm(stream, total=len(gtasks), desc=f'extracting {label}', unit='box'):
                    if r is not None:
                        _grp, h, source = r
                        sources[h] = source
                if not sources:
                    print('\033[96m' + f'{label}: 0 training boxes extracted. Skipping export.' + '\033[0m')
                    continue
                print('\033[96m' + f'{label}: {len(sources)} training boxes{_flavour_detail(sources)} - saving as {out_path}' + '\033[0m')
                se_scnt.pack_staging_dir(os.path.join(staging_root, group_key),
                                         os.path.join(output_directory, out_path),
                                         apix=apix_final, features=feats_list, sources=sources,
                                         z_jitter=Z_JITTER)
                shutil.rmtree(os.path.join(staging_root, group_key), ignore_errors=True)
        finally:
            if pool is not None:
                pool.close()
                pool.join()
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)

