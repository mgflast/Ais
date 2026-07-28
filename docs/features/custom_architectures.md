# Custom model architectures

Ais loads its network architectures at startup — the ones that ship with it, plus any you add yourself. Your own architectures go in the **user models directory**:

- **Windows:** `C:\Users\<you>\.Ais\models`
- **Linux:** `~/.Ais/models`

(The built-in models live inside the installed package, which is often read-only — on a cluster, for example — so your own models go in this user directory instead.) You can drop a `.py` file there directly, or install one through the GUI: **Settings → Model settings → Model library → Install a model** opens a file browser and copies the `.py` in for you. The same menu lists the models you have installed and lets you reload or delete them.

Every model file defines three things:

- `title` — the name shown in the model dropdown.
- `include` — a boolean; set it to `False` to keep the model out of the GUI.
- `create(input_shape)` — a function returning the model object. `input_shape` is `(box_size, box_size, box_depth)`.

## Keras models

Most built-in models are plain `tensorflow.keras` models, which is what Ais expects by default. `create` builds one and returns it, compiled:

```python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D
from tensorflow.keras.optimizers import Adam

title = "My model"   # shown in the model dropdown
include = True       # set to False to hide it

def create(input_shape):
    inputs = Input(input_shape)
    x = Conv2D(32, 3, activation="relu", padding="same")(inputs)
    outputs = Conv2D(1, 1, activation="sigmoid", padding="same")(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(), loss="binary_crossentropy")
    return model
```

The output must be a single-channel map with the same width and height as the input, and values in 0–1 (a `sigmoid` activation). The built-in architectures are written the same way — you can read them in the [`Ais/models`](https://github.com/mgflast/Ais/tree/master/Ais/models) directory of the source.

## Non-Keras models

`create` can return any object, not just a Keras model. Ais drives it exactly as it drives a Keras model, so the object has to implement the methods Ais calls, with matching signatures:

- `count_params()` — the parameter count (inference parameters only; a GAN, for instance, would exclude its discriminator).
- `fit(...)` — called as `keras.Model.fit` is: a data generator plus `steps_per_epoch`, `validation_data`, `epochs`, and `callbacks`.
- `predict(images)` — inference on a batch of boxes.
- `save(path)` — writes the model. Ais reloads it later with Keras's `load_model`, so `save` must produce a Keras-loadable file.

For a real example, see [`pix2pix.py`](https://github.com/mgflast/Ais/blob/master/Ais/models/pix2pix.py): it wraps a generator and a discriminator in one class for training, and saves the generator as a plain Keras model for inference.
