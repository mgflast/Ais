"""Keras 2 / Keras 3 compatibility for Ais models.

TensorFlow 2.8-2.15 ship Keras 2 as tf.keras; TensorFlow 2.16 and later ship Keras 3 (Keras 2 stays available as
the tf-keras package, with TF_USE_LEGACY_KERAS=1). Ais imports Keras only through tensorflow.keras, so it always
uses the Keras that TensorFlow uses, and everything here works under both.

The model file inside a .scnm is a full-model .h5 in the format of the Keras that saved it. Keras 3 reads a Keras 2
file once its transposed convolutions accept the 'groups' argument Keras 2 wrote (TRANSPOSE_COMPAT). Keras 2 cannot
parse the model description in a Keras 3 file, but both store the weights in the same layout, so load_keras_model
can rebuild the architecture and read only the weights.
"""

import importlib
import json
import os
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2DTranspose, Conv3DTranspose
from tensorflow.keras.models import Model, load_model, clone_model

# TF 2.15's tf.keras and tf-keras leave the version to their packages (keras, tf_keras)
KERAS_VERSION = getattr(tf.keras, "__version__", None) or importlib.import_module(Model.__module__.split(".")[0]).__version__
KERAS_3 = int(KERAS_VERSION.split(".")[0]) >= 3


def _accepting_groups(base):
    """base, accepting the 'groups' argument that Keras 2 writes into a transposed convolution's config. Keras 2's
    transposed convolutions ignore it and Keras 3's have no such argument, so only groups=1 is accepted."""

    class Layer(base):
        def __init__(self, *args, groups=1, **kwargs):
            if groups not in (1, None):
                raise ValueError(f"{base.__name__} with groups={groups} is not supported.")
            super().__init__(*args, **kwargs)

    Layer.__name__ = Layer.__qualname__ = base.__name__
    return Layer


TRANSPOSE_COMPAT = {"Conv2DTranspose": _accepting_groups(Conv2DTranspose),
                    "Conv3DTranspose": _accepting_groups(Conv3DTranspose)}


def load_keras_model(path, rebuild=None, compile=False):
    """Load the full-model .h5 file at path. If its model description cannot be read (a model saved by Keras 3,
    opened in Keras 2) and rebuild is given, build the model with rebuild() and read only the weights from the file.
    If that fails too, the error names both causes."""
    try:
        return load_model(path, compile=compile, custom_objects=TRANSPOSE_COMPAT)
    except Exception as e:
        if rebuild is None:
            raise
        reason = f"{type(e).__name__}: {(str(e).strip().splitlines() or [''])[0][:200]}"
        try:
            model = rebuild()
            model.load_weights(path)
        except Exception as e_rebuild:
            raise RuntimeError(f"could not load {os.path.basename(path)} ({reason}), and rebuilding the architecture "
                               f"to read only its weights failed too ({type(e_rebuild).__name__}: {e_rebuild})") from e
        print(f"Note: rebuilt the architecture to read the weights in {os.path.basename(path)}, since its model "
              f"description could not be read with Keras {KERAS_VERSION} ({reason}).")
        return model


def save_keras_model(model, path):
    """model.save(path), as a full-model .h5 file in the format of the Keras in use. Under Keras 2, the 'optional'
    that tf-keras (Keras 2 for TensorFlow 2.16 and later) writes into an InputLayer's config is dropped while it is
    false, because Keras 2.15 and older refuse the argument."""
    model.save(path)
    if KERAS_3:
        return
    import h5py
    with h5py.File(path, "r+") as f:
        config = f.attrs.get("model_config")
        if config is None:
            return
        config = json.loads(config.decode("utf-8") if hasattr(config, "decode") else config)
        changed = False
        for layer in config.get("config", {}).get("layers", []):
            if layer.get("class_name") == "InputLayer" and layer["config"].get("optional") is False:
                del layer["config"]["optional"]
                changed = True
        if changed:
            f.attrs["model_config"] = json.dumps(config).encode("utf-8")


def with_free_input(model, batch=None):
    """A copy of model, with the same weights, whose input takes any Y and X size. The rest of the input shape (the
    depth, and the channel axis of a 3D model) is kept; batch sets the batch size, None for any."""
    inp = Input(shape=(None, None) + tuple(model.input_shape[3:]), batch_size=batch)
    # Keras 3's clone_model also compiles the copy of a compiled model, from a config that cannot name the
    # architectures' loss functions; the copy is for inference and is not compiled (as in Keras 2)
    compiled = getattr(model, "compiled", False)
    if compiled:
        model.compiled = False
    try:
        if not isinstance(model.input, list):
            clone = clone_model(model, input_tensors=inp)
        else:
            # the architectures pass their input as a one-element list; Keras 3 keeps that structure, and
            # clone_model requires it, but then warns on every call with a single array. So the copy takes one.
            clone = clone_model(model, input_tensors=[inp])
            clone = Model(inp, clone.outputs[0] if len(clone.outputs) == 1 else clone.outputs, name=clone.name)
    finally:
        if compiled:
            model.compiled = True
    clone.set_weights(model.get_weights())
    return clone


def no_auto_xla(model):
    """Keras 3 XLA-compiles fit() and predict() on one GPU without a distribution strategy (jit_compile='auto');
    Keras 2 never does. fit() sees one input shape and gains from it, but predict() compiles again for every new
    input shape, as the GUI's slices and views have. So build model's predict function without XLA, leaving fit()
    as it is, and return model."""
    if KERAS_3 and isinstance(model, Model) and model.jit_compile:
        model.jit_compile = False
        model.make_predict_function(force=True)
        model.jit_compile = True
    return model
