"""
Training data audit tool.

Runs a trained model on its own training data and measures per-sample
precision, recall, and Dice at threshold 0.5. Saves arrays and metrics
to disk, then launches a Streamlit browser app for interactive inspection.

Usage (via ais CLI, gated):
    ais audit -m model.scnm -d training.scnt [-o audit_output]
    ais audit -m model.scnm -d training.scnt --skip
"""

import os, sys, json, tempfile, tarfile, glob, csv, subprocess
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tifffile
import Ais.core.se_scnt as se_scnt


def load_model_and_data(model_path, data_path, gpu_id=0):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    for device in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(device, True)
    from keras.models import load_model
    from keras.models import Model
    from keras.backend import clear_session

    with tempfile.TemporaryDirectory() as tmp:
        with tarfile.open(model_path, 'r') as archive:
            archive.extractall(path=tmp)
        weights_file = glob.glob(os.path.join(tmp, "*_weights.h5"))[0]
        metadata_file = glob.glob(os.path.join(tmp, "*_metadata.json"))[0]
        model = load_model(weights_file, compile=False)
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

    # Rebuild model with flexible spatial dims (same as se_model.toggle_inference)
    config = model.get_config()
    weights = model.get_weights()
    input_shape = list(config["layers"][0]["config"]["batch_input_shape"])
    input_shape[0] = 1
    input_shape[1] = None
    input_shape[2] = None
    config["layers"][0]["config"]["batch_input_shape"] = tuple(input_shape)
    del model
    clear_session()
    model = Model.from_config(config)
    model.set_weights(weights)

    model_title = metadata.get('title', 'unknown')
    model_depth = metadata.get('model_depth', 1)
    print(f"Model: {model_title} (depth={model_depth}, apix={metadata.get('apix', '?')})")

    ts = se_scnt.open_training_set(data_path)
    n = ts.n_samples
    x = np.zeros((n, ts.box_shape, ts.box_shape, ts.box_depth), dtype=np.float32)
    y = np.zeros((n, ts.box_shape, ts.box_shape), dtype=np.float32)
    for i in range(n):
        xi, yi = ts.get_sample(i, training=False)  # annotated flavour, no mixing
        x[i] = xi
        y[i] = yi[:, :, 0]
    source_records = ts.source_records()
    ts.close()

    print(f"Training data: {len(x)} samples, {x.shape[1]}x{x.shape[2]}x{x.shape[3]}")
    return model, x, y, model_depth, model_title, source_records


def run_predictions(model, x, model_depth):
    """Run inference with reflect-padding so edges are handled properly.
    Pad to next multiple of 32, with at least 32px on each side."""
    n = len(x)
    h, w = x.shape[1], x.shape[2]
    pad = 32
    padded_h = int(np.ceil((h + 2 * pad) / 32) * 32)
    padded_w = int(np.ceil((w + 2 * pad) / 32) * 32)
    pad_top = (padded_h - h) // 2
    pad_bot = padded_h - h - pad_top
    pad_left = (padded_w - w) // 2
    pad_right = padded_w - w - pad_left

    predictions = np.zeros((n, h, w), dtype=np.float32)
    input_rank = len(model.input_shape)

    for i in range(n):
        inp = x[i:i+1].astype(np.float32)
        # Normalize to zero-mean unit-variance (same as training and inference)
        mu = np.mean(inp)
        std = np.std(inp) + 1e-7
        inp = (inp - mu) / std
        # Pad spatial dims: inp is (1, H, W, D)
        inp = np.pad(inp, ((0, 0), (pad_top, pad_bot), (pad_left, pad_right), (0, 0)), mode='reflect')
        if input_rank == 5:
            inp = inp[..., np.newaxis]
        pred = model(inp, training=False).numpy()
        pred = np.squeeze(pred)
        # Crop back to original size
        predictions[i] = pred[pad_top:pad_top + h, pad_left:pad_left + w]
        if (i + 1) % 50 == 0 or i == n - 1:
            print(f"  predicted {i + 1}/{n}")

    return predictions


def compute_metrics(predictions, labels, threshold=0.5):
    results = []
    for i in range(len(predictions)):
        pred_bin = (predictions[i] > threshold).astype(bool)
        pos_mask = labels[i] == 1
        neg_mask = labels[i] == 0
        valid = pos_mask | neg_mask

        if not valid.any():
            results.append({'idx': i, 'precision': float('nan'), 'recall': float('nan'), 'dice': float('nan'), 'n_pos': 0, 'is_positive': False})
            continue

        pred_valid = pred_bin[valid]
        label_valid = pos_mask[valid]
        tp = (pred_valid & label_valid).sum()
        fp = (pred_valid & ~label_valid).sum()
        fn = (~pred_valid & label_valid).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
        recall = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
        dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else float('nan')

        # For negative samples (no foreground label), report FP rate instead of Dice
        is_pos = pos_mask.any()
        fp_rate = float(fp / neg_mask.sum()) if (not is_pos and neg_mask.sum() > 0) else float('nan')

        results.append({
            'idx': i, 'precision': precision, 'recall': recall, 'dice': dice,
            'n_pos': int(pos_mask.sum()), 'is_positive': is_pos, 'fp_rate': fp_rate
        })
    return results


def _run_processing(model_path, data_path, gpu_id, out_dir, sources_path=None):
    os.makedirs(out_dir, exist_ok=True)

    model, x, labels, model_depth, model_title, source_records = load_model_and_data(model_path, data_path, gpu_id)
    predictions = run_predictions(model, x, model_depth)
    metrics = compute_metrics(predictions, labels)

    np.savez_compressed(os.path.join(out_dir, 'audit_data.npz'), x=x, labels=labels, predictions=predictions)

    csv_path = os.path.join(out_dir, 'audit_metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['idx', 'precision', 'recall', 'dice', 'n_pos', 'is_positive', 'fp_rate'])
        writer.writeheader()
        for m in metrics:
            writer.writerow(m)

    # Sources: prefer the records embedded in the new-format .scnt, else fall back
    # to a provided/auto-detected legacy _sources.star sidecar.
    if any(source_records):
        import starfile, pandas as pd
        df = pd.DataFrame(source_records)
        starfile.write({'sources': df}, os.path.join(out_dir, 'audit_sources.star'), overwrite=True)
    elif sources_path and os.path.exists(sources_path):
        import shutil
        shutil.copy2(sources_path, os.path.join(out_dir, 'audit_sources.star'))

    with open(os.path.join(out_dir, 'audit_meta.json'), 'w') as f:
        json.dump({'model_title': model_title, 'model_path': model_path, 'data_path': data_path, 'n_samples': len(x)}, f)

    # Print summary
    dices = [m['dice'] for m in metrics if m['is_positive'] and not np.isnan(m['dice'])]
    positives = [m for m in metrics if m['is_positive']]
    negatives = [m for m in metrics if not m['is_positive']]
    fp_negatives = [m for m in negatives if not np.isnan(m['fp_rate']) and m['fp_rate'] > 0]
    print(f"\n--- Audit summary for '{model_title}' ---")
    print(f"  {len(metrics)} samples total ({len(positives)} positive, {len(negatives)} negative)")
    if dices:
        print(f"  Dice (positive samples): mean={np.mean(dices):.3f}  median={np.median(dices):.3f}  min={np.min(dices):.3f}")
    precisions = [m['precision'] for m in metrics if not np.isnan(m['precision'])]
    recalls = [m['recall'] for m in metrics if not np.isnan(m['recall'])]
    if precisions:
        print(f"  Precision: mean={np.mean(precisions):.3f}")
    if recalls:
        print(f"  Recall:    mean={np.mean(recalls):.3f}")
    bad = [m for m in metrics if m['is_positive'] and not np.isnan(m['dice']) and m['dice'] < 0.3]
    if bad:
        print(f"  {len(bad)} positive samples with Dice < 0.3")
    if fp_negatives:
        print(f"  {len(fp_negatives)}/{len(negatives)} negative samples with false positive predictions")
    print(f"\nResults saved to {out_dir}/")


STREAMLIT_APP = r'''
import os, json, sys
import numpy as np
import pandas as pd
import streamlit as st
import starfile
from PIL import Image, ImageDraw, ImageFont

out_dir = sys.argv[-1]

@st.cache_data
def load_data(out_dir):
    data = np.load(os.path.join(out_dir, 'audit_data.npz'))
    metrics = pd.read_csv(os.path.join(out_dir, 'audit_metrics.csv'))
    with open(os.path.join(out_dir, 'audit_meta.json')) as f:
        meta = json.load(f)
    sources = None
    src_path = os.path.join(out_dir, 'audit_sources.star')
    if os.path.exists(src_path):
        star_data = starfile.read(src_path)
        sources = star_data['sources'] if isinstance(star_data, dict) else star_data
    return data['x'], data['labels'], data['predictions'], metrics, meta, sources

x, labels, predictions, metrics_df, meta, sources_df = load_data(out_dir)

GAP = 2  # pixels between tiles

def make_tile(i):
    """Compose 4 views into a single 2x2 image for sample i."""
    mid = x.shape[3] // 2
    inp_slice = x[i, :, :, mid]
    label_slice = labels[i]
    pred_slice = predictions[i]
    h, w = inp_slice.shape

    inp_norm = inp_slice - inp_slice.min()
    if inp_norm.max() > 0:
        inp_norm = inp_norm / inp_norm.max()
    inp_img = (inp_norm * 255).astype(np.uint8)

    label_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    label_rgb[label_slice == 1] = [0, 200, 0]
    label_rgb[label_slice == 0] = [30, 30, 30]
    label_rgb[label_slice == 2] = [80, 80, 80]

    pred_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    pred_rgb[:, :, 0] = (np.clip(pred_slice, 0, 1) * 255).astype(np.uint8)

    pred_bin = pred_slice > 0.5
    pos_mask = label_slice == 1
    neg_mask = label_slice == 0
    tp = pred_bin & pos_mask
    fp = pred_bin & neg_mask
    fn = ~pred_bin & pos_mask
    error_rgb = np.stack([inp_img] * 3, axis=-1)
    error_rgb[tp] = [0, 200, 0]
    error_rgb[fp] = [255, 50, 50]
    error_rgb[fn] = [50, 50, 255]

    inp_rgb = np.stack([inp_img] * 3, axis=-1)

    # 2x2 grid with thin gap
    canvas = np.full((2 * h + GAP, 2 * w + GAP, 3), 40, dtype=np.uint8)
    canvas[:h, :w] = inp_rgb
    canvas[:h, w + GAP:] = pred_rgb
    canvas[h + GAP:, :w] = label_rgb
    canvas[h + GAP:, w + GAP:] = error_rgb

    # Draw tiny labels
    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)
    for txt, pos in [("input", (2, 1)), ("prediction", (w + GAP + 2, 1)),
                     ("label", (2, h + GAP + 1)), ("error", (w + GAP + 2, h + GAP + 1))]:
        draw.text(pos, txt, fill=(255, 255, 255))
    return np.array(img)


def _get_cmd_path():
    """Find the pom_to_ais.cmd path. Try reading POM_COMMAND_DIR from Ais settings."""
    ais_settings = os.path.join(os.path.expanduser("~"), ".Ais", "settings.txt")
    cmd_dir = os.path.join(os.path.expanduser("~"), ".Ais")
    if os.path.exists(ais_settings):
        try:
            import json as _json
            with open(ais_settings) as f:
                s = _json.load(f)
            d = s.get("POM_COMMAND_DIR", "")
            if d and os.path.isdir(d):
                cmd_dir = d
        except:
            pass
    return os.path.join(cmd_dir, "pom_to_ais.cmd")

def _to_windows_path(p):
    """Map /cephfs/mlast paths to Z: drive and use backslashes."""
    if p.startswith('/cephfs/mlast'):
        p = 'Z:' + p[len('/cephfs/mlast'):]
    return p.replace('/', '\\')

def open_in_ais(tomo_path):
    cmd_path = _get_cmd_path()
    os.makedirs(os.path.dirname(cmd_path), exist_ok=True)
    win_path = _to_windows_path(tomo_path)
    scns_path = win_path.replace('.mrc', '.scns')
    # Write the Windows path so Ais on Windows can open it
    with open(cmd_path, 'a') as f:
        f.write(f"open\t{scns_path}\n")


def sample_caption(idx):
    i = int(metrics_df.loc[idx, 'idx'])
    m = metrics_df.loc[idx]
    is_pos = m['is_positive']
    tag = "pos" if is_pos else "neg"
    if is_pos:
        d = f"{m['dice']:.3f}" if not pd.isna(m['dice']) else "n/a"
        p = f"{m['precision']:.3f}" if not pd.isna(m['precision']) else ""
        r = f"{m['recall']:.3f}" if not pd.isna(m['recall']) else ""
        stats = f"D={d} P={p} R={r}"
    else:
        fp_rate = m.get('fp_rate', float('nan'))
        stats = f"FP={fp_rate:.4f}" if (not pd.isna(fp_rate) and fp_rate > 0) else "clean"

    source = ""
    if sources_df is not None and i < len(sources_df):
        row = sources_df.iloc[i]
        source = str(row.get('aisTomogramName', ''))

    return f"#{i} ({tag}) {stats}" + (f" | {source}" if source else "")


# ── Page config ──
st.set_page_config(layout="wide", page_title=f"Audit: {meta['model_title']}")
st.title(f"Training data audit: {meta['model_title']}")
st.caption(f"{meta['n_samples']} samples | {meta['model_path']} | {meta['data_path']}")

# Summary
pos_df = metrics_df[metrics_df['is_positive'] == True]
neg_df = metrics_df[metrics_df['is_positive'] == False]
valid_pos = pos_df.dropna(subset=['dice'])
neg_fp = neg_df[neg_df['fp_rate'] > 0] if 'fp_rate' in neg_df.columns else pd.DataFrame()
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Samples", f"{len(pos_df)} pos / {len(neg_df)} neg")
c2.metric("Mean Dice (pos)", f"{valid_pos['dice'].mean():.3f}" if len(valid_pos) else "n/a")
c3.metric("Median Dice (pos)", f"{valid_pos['dice'].median():.3f}" if len(valid_pos) else "n/a")
c4.metric("Dice < 0.3 (pos)", len(valid_pos[valid_pos['dice'] < 0.3]))
c5.metric("Neg w/ FP", len(neg_fp))

# Controls row
ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([2, 2, 1, 1])
sort_options = ["dice (worst first)", "dice (best first)", "sample index"]
if 'fp_rate' in metrics_df.columns:
    sort_options.append("FP rate (worst first)")
sort_by = ctrl1.selectbox("Sort by", sort_options, index=0)
filter_type = ctrl2.selectbox("Filter", ["All", "Positive only", "Negative only", "Dice < 0.3 (pos)", "Neg with FP"])
cols_per_row = ctrl3.selectbox("Columns", [2, 3, 4, 5], index=1)
per_page = ctrl4.selectbox("Per page", [12, 24, 48, 96], index=1)

if sort_by == "dice (worst first)":
    order = metrics_df.sort_values('dice', ascending=True, na_position='first').index.tolist()
elif sort_by == "dice (best first)":
    order = metrics_df.sort_values('dice', ascending=False, na_position='last').index.tolist()
elif sort_by == "FP rate (worst first)":
    order = metrics_df.sort_values('fp_rate', ascending=False, na_position='last').index.tolist()
else:
    order = list(range(len(metrics_df)))

if filter_type == "Positive only":
    order = [i for i in order if metrics_df.loc[i, 'is_positive']]
elif filter_type == "Negative only":
    order = [i for i in order if not metrics_df.loc[i, 'is_positive']]
elif filter_type == "Dice < 0.3 (pos)":
    order = [i for i in order if metrics_df.loc[i, 'is_positive'] and not pd.isna(metrics_df.loc[i, 'dice']) and metrics_df.loc[i, 'dice'] < 0.3]
elif filter_type == "Neg with FP":
    order = [i for i in order if not metrics_df.loc[i, 'is_positive'] and not pd.isna(metrics_df.loc[i, 'fp_rate']) and metrics_df.loc[i, 'fp_rate'] > 0]

n_pages = max(1, (len(order) + per_page - 1) // per_page)
page_col1, page_col2 = st.columns([4, 1])
page_col1.write(f"**{len(order)}** samples")
page = page_col2.number_input("Page", 1, n_pages, 1) - 1
page_indices = order[page * per_page : (page + 1) * per_page]

# ── Render grid ──
for row_start in range(0, len(page_indices), cols_per_row):
    row_indices = page_indices[row_start:row_start + cols_per_row]
    cols = st.columns(cols_per_row)
    for col_idx, idx in enumerate(row_indices):
        i = int(metrics_df.loc[idx, 'idx'])
        tile = make_tile(i)
        caption = sample_caption(idx)

        with cols[col_idx]:
            st.image(tile, caption=caption, use_container_width=True)
            tomo_path = None
            if sources_df is not None and i < len(sources_df):
                val = sources_df.iloc[i].get('aisTomogramPath', '')
                if pd.notna(val) and str(val).strip():
                    tomo_path = str(val)
            if tomo_path:
                if st.button(":material/open_in_new: Ais", key=f"ais_{i}"):
                    open_in_ais(tomo_path)
'''


def _launch_app(out_dir):
    app_path = os.path.join(out_dir, '_audit_app.py')
    with open(app_path, 'w') as f:
        f.write(STREAMLIT_APP)
    print(f"Launching Streamlit app...")
    subprocess.run([sys.executable, '-m', 'streamlit', 'run', app_path, '--', out_dir])


def run_audit(model_path, data_path, gpu_id=0, out_dir='audit_output', skip=False):
    # New-format .scnt files embed their sources (sources.json); legacy TIFF .scnt
    # files may have a _sources.star sidecar, which we auto-detect as a fallback.
    sources = data_path.replace('.scnt', '_sources.star')
    sources = sources if os.path.exists(sources) else None
    if sources:
        print(f"Found legacy sources sidecar: {sources}")

    if skip:
        npz = os.path.join(out_dir, 'audit_data.npz')
        if not os.path.exists(npz):
            print(f"Error: --skip requested but {npz} not found. Run without --skip first.")
            sys.exit(1)
        print(f"Skipping processing, using existing results in {out_dir}/")
    else:
        _run_processing(model_path, data_path, gpu_id, out_dir, sources)

    # Always copy sources into output dir (even with --skip, user may provide -s)
    if sources and os.path.exists(sources):
        import shutil
        shutil.copy2(sources, os.path.join(out_dir, 'audit_sources.star'))

    _launch_app(out_dir)
