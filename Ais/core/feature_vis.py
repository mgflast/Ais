"""
Feature visualization for Ais segmentation models.

Synthesizes input images that maximally activate the model's output feature, by
gradient ascent on the input (activation maximization; Erhan et al. 2009, Olah et
al. 2017). The Ais default models are fully convolutional.

What is visualized: the model's OUTPUT neuron - the pre-sigmoid logit of the final
layer (optimizing the logit rather than the saturating probability keeps gradients
alive) - synthesized `-n` times, each from a different random init. Each image is
a different "facet" of the density the network reads as the feature; together they
show the range of patterns that maximize the output. (For a multi-output model,
every output channel is visualized `-n` times.)

Objective (`-obj`):
  * 'channel' (default) - maximize the output's MEAN activation over the whole
    image (Olah et al.'s channel objective). No spatial position is privileged,
    so the whole image becomes uniform interpretable texture with no bright/dark
    "centre" hotspot. This is the standard feature-visualization objective.
  * 'neuron' - maximize only the output's centre pixel, so the feature lands at
    the centre of the image (at the cost of a prominent centre).

Crucially, every forward pass during synthesis uses the *same* inference the rest
of Ais uses (SEModel.apply_to_slice): the image is normalized and reflect-padded
before it enters the network. Without that border context the 'same' convolutions
zero-pad the edges and the network barely responds - so an un-padded forward pass
optimizes an activation the deployed model never actually produces. The channel
objective averages over only the original-image region of that padded map (what
Ais keeps at inference), so the reflected padding does not over-weight the border.

To turn raw high-frequency gradient noise into interpretable structure we use a
natural-image prior with fixed, sensible defaults (no CLI knobs):

  * spectral (Fourier) parametrization - the image is optimized as decorrelated
    frequency coefficients scaled by 1/frequency, so gradient descent naturally
    favors low-frequency, less noisy structure (Olah et al. 2017). That 1/f prior
    is high-passed below ~24 px wavelength (a smooth order-2 window; the hidden
    --highpass knob) to keep the result interesting mid/high-frequency texture: it
    stops a trivial image-scale brightness pattern (which for the channel objective
    directly raises the mean, and for the neuron objective is the over-bright/dark
    centre) from being a cheap solution, while leaving the fine texture untouched;
  * transformation robustness - each step the decoded image is randomly scaled
    and jittered before the forward pass, so the objective is averaged over
    transforms and the result is robust rather than adversarial;
  * in-distribution input - each step the image is normalized to mean 0 / std 1,
    exactly like the training/inference preprocessing, so the optimizer cannot
    "win" by inflating absolute contrast and the image stays in the regime the
    network was trained on.

Note on cryo-ET specifically: these models are per-pixel segmentation nets
trained on low-SNR tomograms, so even a well-regularized maximization looks like
organized density texture, not a clean "platonic" shape - the network genuinely
keys on local texture. For a real-example view, rank training patches by
activation instead (dataset mode).

Multi-GPU: neurons are independent optimization problems, so the selected
channels are sharded round-robin across the GPUs passed to `-gpu` (one worker
process per GPU, matching the `segment` command). Each worker writes its
per-neuron results to a staging directory as it finishes them; the parent
process - which never imports TensorFlow, so it never initializes CUDA - shows a
progress bar by counting those files and then assembles the combined outputs.
"""

import os
import json
import time
import random
import shutil
import tempfile
import multiprocessing

import numpy as np
import mrcfile
from tqdm import tqdm


# Fixed synthesis hyperparameters (not exposed on the CLI - see module docstring).
_LEARNING_RATE = 0.05
_INIT_STDDEV = 0.01
_SCALE_JITTER = 0.05    # random rescale range each step, +/- fraction
_CHUNK = 16             # max images optimized per batch
# Target upper bound on (batch x padded-pixels) so a batch fits GPU memory: the
# batch shrinks as -size grows. Calibrated for ~8 GB; an OOM-retry halves it further
# for heavy architectures. e.g. ~8 images at 128 px, ~2 at 256 px.
_BATCH_PIXEL_BUDGET = 300_000
_DOWNSAMPLE_MULTIPLE = 32  # -size is rounded up to a multiple of this
# Fourier-prior high-pass cutoff WAVELENGTH (px): structures coarser than this - the
# image-scale brightness dome that over-brightens/darkens the centre - are rolled off. 0 disables.
_HIGHPASS_CUTOFF_PX = 24.0


def _round_size(size):
    # Floor at 2x the downsample multiple (64): the reflect-pad that matches Ais
    # inference adds 32 px per side, and REFLECT padding must be < the dimension.
    size = max(int(size), 2 * _DOWNSAMPLE_MULTIPLE)
    if size % _DOWNSAMPLE_MULTIPLE == 0:
        return size
    return ((size // _DOWNSAMPLE_MULTIPLE) + 1) * _DOWNSAMPLE_MULTIPLE


def _center_channels(act):
    """act: (b, *spatial, C) -> (b, C): the center pixel of every channel.

    The objective is this single center pixel (not the spatial mean), so the
    feature of interest is guaranteed to be present at the centre of the
    synthesized image rather than only somewhere in it.
    """
    idx = [slice(None)] + [int(s) // 2 for s in act.shape[1:-1]] + [slice(None)]
    return act[tuple(idx)]


def _fourier_scale(h, w, decay=1.0, highpass_cutoff_px=0.0):
    """Per-frequency scaling for the spectral image parametrization (rfft2d layout,
    last axis is w//2 + 1 non-negative frequencies).

    The base 1/frequency weighting favours natural, low-frequency structure - but it
    makes an image-scale brightness "dome" (DC + the lowest one or two modes) the
    single cheapest thing gradient ascent can build, which is exactly the
    over-bright/dark centre of activation-maximization images. When
    highpass_cutoff_px > 0 we multiply the 1/f scale by a smooth order-2 high-pass
    window W(f) = f^2 / (f^2 + fc^2), fc = 1/cutoff (cycles/px): W(0) = 0 removes DC,
    the dome modes are cut ~3-10x, and W -> 1 for f >> fc so the mid/high-frequency
    texture keeps its natural weighting. The cutoff is a fixed pixel wavelength, so
    it tracks the network's (fixed-pixel) receptive field at any -size.
    """
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[:w // 2 + 1][None, :]
    freqs = np.sqrt(fy ** 2 + fx ** 2)
    scale = 1.0 / np.maximum(freqs, 1.0 / max(h, w)) ** decay
    scale *= np.sqrt(h * w)
    if highpass_cutoff_px and highpass_cutoff_px > 0:
        fc = 1.0 / float(highpass_cutoff_px)                    # cutoff in cycles/pixel
        scale = scale * (freqs ** 2 / (freqs ** 2 + fc ** 2))   # order-2 high-pass; W(0)=0
    return scale.astype(np.float32)


def _ais_padding(h, w):
    """Reflect-pad amounts matching SEModel.apply_to_slice (multiple of 32, +64)."""
    ph = ((32 - h % 32) % 32) + 64
    pw = ((32 - w % 32) % 32) + 64
    return ph, pw


def _synthesize(tf, extractor, channels, input_shape, iterations, tag="", save_fn=None, heartbeat=None,
                objective='channel', normalize=True, highpass_px=_HIGHPASS_CUTOFF_PX, labels=None):
    """Activation maximization with a natural-image prior, batched over `channels`.

    The image is parametrized in decorrelated Fourier space (low frequencies
    favored) and, each step, randomly scaled + jittered and then normalized to
    mean 0 / std 1 before the forward pass. Channels in a chunk are optimized
    together via the diagonal trick: image j is scored only by its own channel,
    and since image j only influences activation j, each image gets exactly its
    own channel's gradient (the shared per-step transform and per-image
    normalization preserve this).

    `objective`:
      * 'channel' (default) - maximize the channel's MEAN activation over all
        spatial positions (Olah et al.'s channel objective). No position is
        privileged, so the whole image becomes uniform texture with no
        bright/dark-centre hotspot.
      * 'neuron' - maximize only the channel's centre pixel, so the feature lands
        at the centre (at the cost of a prominent centre).

    Every forward pass reflect-pads the image (via `extractor`, which must accept
    variable spatial size) to match Ais's real inference; the padding is symmetric
    so the centre of the padded activation map is the centre of the original image.

    If `save_fn` is given it is called as save_fn(channel, image) for each channel
    as its chunk finishes, so a parent process can track progress by counting the
    files it writes. `heartbeat(partial)` (if given) is called during optimization
    with the number of not-yet-saved neurons currently in progress, so the parent
    can show smooth sub-chunk progress. Also returns an array
    (len(channels), *input_shape).
    """
    h, w = int(input_shape[0]), int(input_shape[1])
    extra = tuple(int(d) for d in input_shape[2:])        # depth dims after H, W
    k = int(np.prod(extra)) if extra else 1               # channels for the 2D FFT
    wf = w // 2 + 1
    jitter = max(1, h // 16)
    ph, pw = _ais_padding(h, w)               # reflect-pad amounts (Ais inference)
    ph0, pw0 = ph // 2, pw // 2
    scale_np = _fourier_scale(h, w, highpass_cutoff_px=highpass_px)
    scale_c = tf.complex(tf.constant(scale_np), tf.constant(np.zeros_like(scale_np)))

    def decode(spectrum):
        sp = tf.complex(spectrum[..., 0], spectrum[..., 1])   # (b, K, H, Wf)
        img = tf.signal.irfft2d(sp * scale_c, fft_length=[h, w])  # (b, K, H, W)
        return tf.transpose(img, [0, 2, 3, 1])                # (b, H, W, K)

    if labels is None:
        labels = list(channels)   # save ids; distinct from `channels` (the gather targets)

    def optimize_chunk(cidx):
        b = len(cidx)
        spectrum = tf.Variable(tf.random.normal((b, k, h, wf, 2), stddev=_INIT_STDDEV))
        opt = tf.keras.optimizers.Adam(_LEARNING_RATE)
        gather_idx = tf.stack([tf.range(b), tf.constant(cidx, tf.int32)], axis=1)
        for it in range(iterations):
            if heartbeat is not None and it % 5 == 0:
                heartbeat(b * it / iterations)
            s = float(np.random.uniform(1.0 - _SCALE_JITTER, 1.0 + _SCALE_JITTER))
            nh, nw = max(4, int(round(h * s))), max(4, int(round(w * s)))
            dy = int(np.random.randint(-jitter, jitter + 1))
            dx = int(np.random.randint(-jitter, jitter + 1))
            with tf.GradientTape() as tape:
                img = decode(spectrum)                         # (b, H, W, K)
                img = tf.image.resize(img, [nh, nw])           # random scale ...
                img = tf.image.resize_with_crop_or_pad(img, h, w)
                img = tf.roll(img, [dy, dx], axis=[1, 2])      # ... and jitter
                if normalize:
                    mean = tf.reduce_mean(img, axis=[1, 2, 3], keepdims=True)
                    std = tf.math.reduce_std(img, axis=[1, 2, 3], keepdims=True)
                    img = (img - mean) / (std + 1e-6)          # normalize, then ...
                img = tf.pad(img, [[0, 0], [ph0, ph - ph0], [pw0, pw - pw0], [0, 0]], mode='REFLECT')
                model_in = tf.reshape(img, [b, h + ph, w + pw, *extra]) if len(extra) > 1 else img
                act = extractor(model_in, training=False)      # ... reflect-pad, exactly like Ais
                if objective == 'channel':
                    # Average over the ORIGINAL-image region of the (padded) map, not the
                    # whole padded map. For a 64 px image the reflect-pad is 75% of the
                    # padded area and mirrors the outer band, so averaging over all of it
                    # over-weights the border (edge features get duplicated into the
                    # padding) and features drift into a ring, leaving the centre empty.
                    # This crop also matches what Ais outputs (it crops back to size).
                    mh, mw = int(act.shape[1]), int(act.shape[2])
                    y0, y1 = int(round(ph0 * mh / (h + ph))), int(round((ph0 + h) * mh / (h + ph)))
                    x0, x1 = int(round(pw0 * mw / (w + pw))), int(round((pw0 + w) * mw / (w + pw)))
                    core = act[:, y0:y1, x0:x1]
                    scores = tf.reduce_mean(core, axis=list(range(1, core.shape.rank - 1)))  # (b, C)
                else:                                          # 'neuron'
                    scores = _center_channels(act)             # channel centre pixel (b, C)
                loss = -tf.reduce_sum(tf.gather_nd(scores, gather_idx))  # maximize activation
            opt.apply_gradients([(tape.gradient(loss, spectrum), spectrum)])
        img = decode(spectrum).numpy().reshape((b, *input_shape))  # clean final render
        for j in range(b):
            v = img[j]
            img[j] = (v - v.mean()) / (v.std() + 1e-7)
        return img

    out = np.zeros((len(channels), *input_shape), dtype=np.float32)
    # Batch size shrinks as the (padded) image grows so a batch fits GPU memory; an
    # OOM halves it further and retries the chunk.
    chunk = max(1, min(_CHUNK, _BATCH_PIXEL_BUDGET // ((h + ph) * (w + pw))))
    start = 0
    while start < len(channels):
        b = min(chunk, len(channels) - start)
        try:
            img = optimize_chunk(list(channels[start:start + b]))
        except tf.errors.ResourceExhaustedError:
            if b <= 1:
                raise
            chunk = max(1, b // 2)
            print(f"{tag}out of GPU memory at batch {b}; retrying with batch {chunk}")
            continue
        out[start:start + b] = img
        if heartbeat is not None:
            heartbeat(0.0)   # this chunk's neurons are now counted via save_fn
        if save_fn is not None:
            for j in range(b):
                save_fn(labels[start + j], img[j])
        else:
            print(f"{tag}synthesized neurons {start + 1}-{start + b} / {len(channels)}")
        start += b
    return out


def _center_slice_stack(imgs):
    """(N, H, W, D[, 1]) -> (N, H, W) taking the central Z slice."""
    arr = imgs
    if arr.ndim == 5:          # (N, H, W, D, 1) 3D model
        arr = arr[..., 0]
    d = arr.shape[-1]          # depth
    return arr[..., d // 2]


def _stretch(img):
    lo, hi = np.percentile(img, (2, 98))
    if hi <= lo:
        hi = lo + 1e-6
    return (np.clip((img - lo) / (hi - lo), 0.0, 1.0) * 255).astype(np.uint8)


def _save_montage(stack2d, labels, path):
    from PIL import Image, ImageDraw
    n, h, w = stack2d.shape
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    pad = 2
    canvas = Image.new('L', (cols * w + (cols + 1) * pad, rows * h + (rows + 1) * pad), color=40)
    draw = ImageDraw.Draw(canvas)
    for i in range(n):
        r, c = divmod(i, cols)
        x0, y0 = pad + c * (w + pad), pad + r * (h + pad)
        canvas.paste(Image.fromarray(_stretch(stack2d[i])), (x0, y0))
        if labels is not None:
            draw.text((x0 + 2, y0 + 1), str(labels[i]), fill=255)
    canvas.save(path)


def _save_mrc(stack2d, apix, path):
    with mrcfile.new(path, overwrite=True) as m:
        m.set_data(np.ascontiguousarray(stack2d, dtype=np.float32))
        if apix and apix > 0:
            m.voxel_size = float(apix)


def _fviz_worker(rank, gpus, model_path, size, n_images, iterations, seed, staging_dir, highpass, objective):
    """One GPU's share of the work: load the model and synthesize this rank's
    shard of the output-neuron images, writing each finished image to staging so
    the parent can count them for the progress bar."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus[rank])

    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    from tensorflow.keras.models import Model, clone_model
    from tensorflow.keras.layers import Input
    from Ais.core.se_model import SEModel

    tf.random.set_seed(seed + rank)
    np.random.seed((seed + rank) % (2 ** 32))

    n_gpus = len(gpus)

    se_model = SEModel(no_glfw=True)
    se_model.load(model_path, compile=False)
    if se_model.model is None:
        print(f"[GPU {gpus[rank]}] could not load model from {model_path}. Aborting.")
        return
    base = se_model.model
    depth = se_model.model_depth
    input_rank = len(base.input_shape)
    input_shape = (size, size, depth) if input_rank == 4 else (size, size, depth, 1)

    # Variable-size, flexible-batch input: flexible batch so we can optimize a
    # batch of neurons at once (the loaded inference model fixes batch=1), and
    # variable spatial size so _synthesize can reflect-pad each forward pass the
    # way Ais's apply_to_slice does.
    var_input = Input(shape=(None, None, depth)) if input_rank == 4 else Input(shape=(None, None, depth, 1))
    model = clone_model(base, input_tensors=var_input)
    model.set_weights(base.get_weights())

    out_layer = model.layers[-1]

    # Output-logit extractor: rebuild the final layer with a linear activation so
    # gradients do not vanish in the sigmoid's saturated regime.
    logit_cfg = out_layer.get_config()
    logit_cfg['activation'] = 'linear'
    logit_cfg['name'] = 'fviz_output_logit'
    linear = out_layer.__class__.from_config(logit_cfg)
    logit = linear(out_layer.input)
    linear.set_weights(out_layer.get_weights())
    n_out = int(logit.shape[-1])

    # Visualize the OUTPUT neuron `n_images` times, each from a different random
    # init - different "facets" of what the network reads as the feature. (For a
    # multi-output model, every output channel is visualized n_images times.)
    jobs = [(oc, i) for oc in range(n_out) for i in range(n_images)]   # (channel, facet)

    if rank == 0:
        # Announce the total up front so the parent's progress bar has a target.
        with open(os.path.join(staging_dir, 'progress.json'), 'w') as f:
            json.dump({'total': len(jobs)}, f)
        with open(os.path.join(staging_dir, 'meta.json'), 'w') as f:
            json.dump({'title': se_model.title, 'architecture': se_model.get_model_title(),
                       'apix': se_model.apix, 'size': size, 'iterations': iterations,
                       'n_out': n_out, 'n_images': n_images, 'jobs': jobs}, f, indent=2)

    out_dir = os.path.join(staging_dir, 'out')
    os.makedirs(out_dir, exist_ok=True)

    def save_fn(gid, img):
        np.save(os.path.join(out_dir, f"{gid}.npy"), _center_slice_stack(img[None])[0])

    hb_path = os.path.join(staging_dir, f"hb_{rank}.json")

    def heartbeat(partial):
        with open(hb_path, 'w') as f:
            json.dump({'partial': float(partial)}, f)

    my = list(enumerate(jobs))[rank::n_gpus]       # [(gid, (channel, facet)), ...]
    if my:
        oe = Model(model.input, logit)
        channels = [oc for _, (oc, _i) in my]      # gather targets: the output channel(s)
        gids = [gid for gid, _ in my]              # unique save ids
        _synthesize(tf, oe, channels, input_shape, iterations, objective=objective,
                    save_fn=save_fn, heartbeat=heartbeat, highpass_px=highpass, labels=gids)
    heartbeat(0.0)


def _count_temp_imgs(staging_dir):
    d = os.path.join(staging_dir, 'out')
    return sum(1 for f in os.listdir(d) if f.endswith('.npy')) if os.path.isdir(d) else 0


def _partial_progress(staging_dir, n_gpus):
    """Sum of the in-progress (not-yet-saved) neuron fractions across workers."""
    s = 0.0
    for r in range(n_gpus):
        p = os.path.join(staging_dir, f"hb_{r}.json")
        if os.path.exists(p):
            try:
                with open(p) as f:
                    s += json.load(f)['partial']
            except Exception:
                pass
    return s


def _assemble(staging_dir, output_directory):
    """Parent-side, TensorFlow-free: gather the per-image .npy files (the output
    neuron synthesized n_images times, in job order) into a combined .mrc stack +
    .png montage and a manifest."""
    meta_path = os.path.join(staging_dir, 'meta.json')
    if not os.path.exists(meta_path):
        print("No results were produced (missing meta.json). Nothing to assemble.")
        return
    with open(meta_path) as f:
        meta = json.load(f)
    title, apix = meta['title'], meta['apix']
    jobs, multi = meta['jobs'], meta['n_out'] > 1

    imgs, labels = [], []
    for gid, (oc, facet) in enumerate(jobs):
        p = os.path.join(staging_dir, 'out', f"{gid}.npy")
        if os.path.exists(p):
            imgs.append(np.load(p))
            labels.append(f"out{oc}.{facet}" if multi else str(facet))

    if not imgs:
        print("No neuron images were produced. Nothing to assemble.")
        return

    stack = np.stack(imgs, axis=0)
    mrc_path = os.path.join(output_directory, f"{title}_feature_visualisation.mrc")
    png_path = os.path.join(output_directory, f"{title}_feature_visualisation.png")
    _save_mrc(stack, apix, mrc_path)
    _save_montage(stack, labels, png_path)

    manifest = {'model': title, 'architecture': meta['architecture'], 'apix': apix,
                'size': meta['size'], 'iterations': meta['iterations'],
                'n_out': meta['n_out'], 'n_images': meta['n_images'],
                'mrc': os.path.basename(mrc_path), 'png': os.path.basename(png_path)}
    with open(os.path.join(output_directory, f"{title}_feature_visualisation.json"), 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {os.path.basename(mrc_path)} and {os.path.basename(png_path)} "
          f"({len(imgs)} output-neuron images). Output in {output_directory}")


def run_feature_visualisation(model_path, size=64, n_images=9, gpu="0",
                              output_directory="feature_visualisation",
                              iterations=400, seed=None, highpass=_HIGHPASS_CUTOFF_PX,
                              objective='channel'):
    gpus = [g.strip() for g in str(gpu).split(',') if g.strip() != '']
    if not gpus:
        gpus = ["0"]

    if not os.path.isabs(model_path):
        model_path = os.path.join(os.getcwd(), model_path)
    if not os.path.isabs(output_directory):
        output_directory = os.path.join(os.getcwd(), output_directory)
    os.makedirs(output_directory, exist_ok=True)

    size = _round_size(int(size))
    if seed is None:
        seed = random.randrange(1 << 30)   # random per run (printed for reproducibility)

    print(f"Feature visualisation: model={os.path.basename(model_path)}, size={size}x{size}, "
          f"n_images={n_images}, objective={objective}, iterations={iterations}, "
          f"GPUs={','.join(gpus)}, highpass={highpass}px, seed={seed}")

    staging_dir = tempfile.mkdtemp(prefix='fviz_')
    try:
        procs = []
        for rank in range(len(gpus)):
            p = multiprocessing.Process(
                target=_fviz_worker,
                args=(rank, gpus, model_path, size, n_images, iterations, seed, staging_dir, highpass, objective),
            )
            p.start()
            procs.append(p)

        # Progress bar driven by counting the per-neuron temp files as workers
        # write them; total is announced by rank 0 once it has loaded the model.
        progress_path = os.path.join(staging_dir, 'progress.json')
        total = None
        last = -1.0
        bar = tqdm(total=None, desc='synthesizing neurons', unit='img', mininterval=0.5)
        try:
            while any(p.is_alive() for p in procs):
                if total is None and os.path.exists(progress_path):
                    try:
                        total = json.load(open(progress_path))['total']
                        bar.total = total
                        bar.refresh()
                    except Exception:
                        pass
                done = _count_temp_imgs(staging_dir)
                cur = min(total, done + _partial_progress(staging_dir, len(gpus))) if total else done
                if cur != last:
                    bar.n = round(cur, 2)
                    bar.refresh()
                    last = cur
                time.sleep(0.5)
            bar.n = _count_temp_imgs(staging_dir)
            if total is not None:
                bar.total = total
            bar.refresh()
        finally:
            bar.close()

        for p in procs:
            p.join()

        _assemble(staging_dir, output_directory)
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)
