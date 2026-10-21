"""Save a Keras model as a Keras 2 full-model .h5 file, under Keras 2 and Keras 3 alike.

Every released Ais (TensorFlow 2.8-2.15, Keras 2) loads the model in a .scnm with load_model(h5, compile=False).
Keras 3 writes the weights of an .h5 in the Keras 2 layout, but describes the model in a schema Keras 2 cannot
parse. Under Keras 3, save_keras2_h5 therefore writes the file itself: the weights in that layout, and the model
description translated to the Keras 2 schema. The translation covers the layers and settings of Ais' built-in
architectures. Anything outside that (a custom layer, a setting Keras 2 does not have) is not guessed at: the
model is saved by Keras 3 instead, with a warning, and released Ais can read it after `ais convert` under Keras 2.
"""

import json
import h5py
import numpy as np
from tensorflow.keras.models import Model
from Ais.core.keras_compat import KERAS_3, KERAS_VERSION, TRANSPOSE_COMPAT, save_keras_model

# The config keys TensorFlow 2.8, 2.10, 2.11 and 2.15 all write, and so all read, per layer class.
_BASE = {"name", "trainable", "dtype"}
_CONV = _BASE | {"filters", "kernel_size", "strides", "padding", "data_format", "dilation_rate", "groups",
                 "activation", "use_bias", "kernel_initializer", "bias_initializer", "kernel_regularizer",
                 "bias_regularizer", "activity_regularizer", "kernel_constraint", "bias_constraint"}
_NORM = _BASE | {"axis", "epsilon", "center", "scale", "beta_initializer", "gamma_initializer",
                 "beta_regularizer", "gamma_regularizer", "beta_constraint", "gamma_constraint"}
_POOL = _BASE | {"pool_size", "strides", "padding", "data_format"}
KERAS2_LAYERS = {
    "InputLayer": {"batch_input_shape", "dtype", "sparse", "ragged", "name"},
    "Conv2D": _CONV, "Conv3D": _CONV,
    "Conv2DTranspose": _CONV | {"output_padding"},
    "Conv3DTranspose": _CONV - {"dilation_rate"} | {"output_padding"},
    "BatchNormalization": _NORM | {"momentum", "moving_mean_initializer", "moving_variance_initializer"},
    "GroupNormalization": _NORM | {"groups"},   # TensorFlow 2.11 and later only
    "MaxPooling2D": _POOL, "MaxPooling3D": _POOL,
    "UpSampling2D": _BASE | {"size", "data_format", "interpolation"},
    "UpSampling3D": _BASE | {"size", "data_format"},
    "Concatenate": _BASE | {"axis"}, "Add": _BASE,
    "Dropout": _BASE | {"rate", "noise_shape", "seed"},
    "Activation": _BASE | {"activation"},
    "ReLU": _BASE | {"max_value", "negative_slope", "threshold"},
    "LeakyReLU": _BASE | {"alpha"},
}
KERAS2_OBJECTS = {"GlorotUniform": {"seed"}, "Zeros": set(), "Ones": set()}
# built-in layers, and the transposed convolutions of a model that was read from a Keras 2 file
_LAYER_MODULES = {"keras.layers", TRANSPOSE_COMPAT["Conv2DTranspose"].__module__}
# keys Keras 3 renamed, and keys only Keras 3 has, which are dropped while they hold the value Keras 2 implies
_RENAMED = {("InputLayer", "batch_shape"): "batch_input_shape", ("LeakyReLU", "negative_slope"): "alpha"}
_IMPLIED = {"optional": False, "synchronized": False, "renorm": False, "renorm_clipping": None,
            "renorm_momentum": 0.99, "dilation_rate": [1, 1, 1], "input_axes": None, "output_axes": None}
_IMPLIED_CALL = {"mask": None, "training": False}


class _Unsupported(Exception):
    pass


def save_keras2_h5(model, path):
    """Save model to path as a full-model .h5 file that Keras 2 can load (uncompiled). Returns False if the model
    had to be saved in the Keras 3 format instead."""
    if not isinstance(model, Model):
        return model.save(path)   # an architecture that saves itself (Pix2pix saves its generator with this function)
    if not KERAS_3:
        save_keras_model(model, path)
        return True
    try:
        if type(model).__name__ != "Functional":
            raise _Unsupported(f"a {type(model).__name__} model")
        config = {"class_name": "Functional", "config": keras2_config(model.get_config())}
    except _Unsupported as e:
        print(f"Warning: the model is saved in the Keras 3 format, as it has {e}, which the Keras 2 writer does not "
              f"cover. Released Ais versions read it only after `ais convert` under Keras 2.")
        model.save(path)
        return False
    with h5py.File(path, "w") as f:
        f.attrs["model_config"] = json.dumps(config).encode()
        group = f.create_group("model_weights")
        for attrs in (f.attrs, group.attrs):
            attrs["backend"] = b"tensorflow"
            attrs["keras_version"] = KERAS_VERSION.encode()
        # Keras 2 matches the weights to the layers by position: the layers in model order, and each layer's
        # trainable weights before its non-trainable ones. Dataset names only need to be unique within a layer.
        group.attrs["layer_names"] = [layer.name.encode() for layer in model.layers]
        for layer in sorted(model.layers, key=lambda layer: layer.name):
            weights = layer.trainable_weights + layer.non_trainable_weights
            g = group.create_group(layer.name)
            g.attrs["weight_names"] = [w.path.encode() for w in weights]
            for w in weights:
                g.create_dataset(w.path, data=w.numpy())
    return True


def keras2_config(config):
    """A Keras 3 functional model config, translated to the Keras 2 schema."""
    config = json.loads(json.dumps(config, default=_json_value))
    if not config.pop("trainable", True):
        raise _Unsupported("a frozen model")
    extra = set(config) - {"name", "layers", "input_layers", "output_layers"}
    if extra:
        raise _Unsupported(f"model setting(s) {sorted(extra)}")
    return {"name": config["name"],
            "layers": [_layer(layer) for layer in config["layers"]],
            # Keras 2 writes a list of [layer, node, tensor] even for a single input or output
            "input_layers": _endpoints(config["input_layers"]),
            "output_layers": _endpoints(config["output_layers"])}


def _layer(layer):
    cls = layer["class_name"]
    if cls not in KERAS2_LAYERS or layer.get("module") not in _LAYER_MODULES:
        raise _Unsupported(f"a {layer.get('module')}.{cls} layer")
    return {"class_name": cls, "config": _settings(cls, layer["config"], KERAS2_LAYERS[cls]),
            "name": layer["name"], "inbound_nodes": [_node(node) for node in layer["inbound_nodes"]]}


def _settings(cls, config, keys):
    out = {}
    for key, value in config.items():
        key = _RENAMED.get((cls, key), key)
        if key not in keys:
            if key in _IMPLIED and value == _IMPLIED[key]:
                continue
            raise _Unsupported(f"{cls} setting {key}={value!r}")
        out[key] = _object(value) if isinstance(value, dict) else value
    return out


def _object(value):
    # Keras 3 writes a dtype policy or an initializer as a dict with module and registered_name; Keras 2 reads a dtype
    # as its name and an initializer as {class_name, config}.
    cls, module = value.get("class_name"), value.get("module")
    if cls in ("DTypePolicy", "FloatDTypePolicy") and module == "keras":
        return value["config"]["name"]
    if cls not in KERAS2_OBJECTS or module != "keras.initializers":
        raise _Unsupported(f"a {cls or value} setting")
    return {"class_name": cls, "config": _settings(cls, value.get("config") or {}, KERAS2_OBJECTS[cls])}


def _node(node):
    # Keras 3: {"args": [tensor or [tensors]], "kwargs": {...}}, each tensor carrying its keras_history
    # [layer, node, tensor]. Keras 2: [[layer, node, tensor, kwargs], ...], one entry per input tensor. Keras 3
    # records mask=None and training=False where Keras 2 records nothing; a training value from the model's own
    # call overrides it, which is what Keras 2 does when the layer is called without one.
    kwargs = {k: v for k, v in node.get("kwargs", {}).items() if _IMPLIED_CALL.get(k, ...) != v}
    if kwargs:
        raise _Unsupported(f"call arguments {kwargs}")
    tensors = []

    def collect(x):
        if isinstance(x, list):
            for item in x:
                collect(item)
        elif isinstance(x, dict) and x.get("class_name") == "__keras_tensor__":
            tensors.append(list(x["config"]["keras_history"]) + [{}])
        else:
            raise _Unsupported(f"a non-tensor layer input {x!r}")

    collect(node["args"])
    return tensors


def _endpoints(endpoints):
    if len(endpoints) == 3 and isinstance(endpoints[0], str):
        endpoints = [endpoints]
    if not all(isinstance(e, list) and len(e) == 3 and isinstance(e[0], str) for e in endpoints):
        raise _Unsupported(f"model inputs or outputs {endpoints}")
    return endpoints


def _json_value(value):
    if isinstance(value, (np.generic, np.ndarray)):
        return value.tolist()
    raise _Unsupported(f"a setting of type {type(value).__name__}")
