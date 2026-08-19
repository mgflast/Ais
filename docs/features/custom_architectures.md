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
- `save(path)` — writes the model.

For loading a saved model back, there are two options. By default Ais reloads it with Keras's `load_model`, so `save` must produce a Keras-loadable file. [`pix2pix.py`](https://github.com/mgflast/Ais/blob/master/Ais/models/pix2pix.py) works this way: it wraps a generator and a discriminator in one class for training, and saves the generator as a plain Keras model for inference. Alternatively, the model file can define a module-level `load_model(path)` function. If it does, Ais calls it instead, and hands it whatever file your object's `save(path)` wrote — which frees the model from Keras entirely and makes other machine learning backends possible.

Two things to be aware of:

- A model saved by an architecture with its own `load_model` only loads on machines where that `.py` (and its backend) is installed — unlike Keras models, the `.scnm` is not self-contained.
- Ais passes Keras callback objects to `fit` for the training progress readout and the stop button. A `fit` that ignores the `callbacks` argument trains fine, but those GUI elements will not respond; to support them, call the callbacks' `on_batch_end`/`on_epoch_end` from your training loop and stop when `stop_training` has been set on your model object.

### Example: VGGNet M in PyTorch

The code below implements the same VGGNet M network that is already available in Ais, but in PyTorch instead of tensorflow.

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

title = "VGGNet M torch"
include = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VGGNetM(nn.Module):
    # the same layers as the Keras VGGNet M, in NCHW
    def __init__(self, in_channels):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(512, 128, 2, stride=2),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 2, stride=2),
            nn.Conv2d(1, 1, 1),
        )

    def forward(self, x):
        return torch.sigmoid(self.dec(self.enc(x)))


def masked_bce(y_pred, y_true, ignore_label=2.0, epsilon=1e-6):
    mask = (y_true != ignore_label).float()
    y_clean = torch.where(y_true == ignore_label, torch.zeros_like(y_true), y_true)
    y_pred = y_pred.clamp(1e-7, 1.0 - 1e-7)
    bce = F.binary_cross_entropy(y_pred, y_clean, reduction="none")
    return (bce * mask).sum() / (mask.sum() + epsilon)


class _LearningRate:
    # Ais sets the rate via model.optimizer.learning_rate.assign(value)
    def __init__(self, opt):
        self._opt = opt

    def assign(self, value):
        for g in self._opt.param_groups:
            g["lr"] = float(value)


class _Optimizer:
    def __init__(self, opt):
        self.learning_rate = _LearningRate(opt)


class TorchWrapper:
    def __init__(self, input_shape):
        h, w, d = input_shape
        self.input_shape = (None, h, w, d)     # rank 4, channels-last, as Ais expects
        self.output_shape = (None, h, w, 1)
        self.net = VGGNetM(d).to(DEVICE)
        self._opt = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        self.optimizer = _Optimizer(self._opt)
        self.loss = masked_bce
        self.stop_training = False

    def compile(self, **kwargs):
        pass   # Ais re-compiles with its own metrics; nothing to do here

    def count_params(self):
        return sum(p.numel() for p in self.net.parameters())

    def _to_torch(self, x):
        if x.ndim == 3:
            x = x[..., np.newaxis]
        return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).permute(0, 3, 1, 2).to(DEVICE)

    def fit(self, generator, steps_per_epoch=1, validation_data=None, validation_steps=None,
            epochs=1, callbacks=None, **kwargs):
        callbacks = callbacks or []
        self.stop_training = False
        for cb in callbacks:
            cb.set_model(self)
            cb.set_params({"steps": steps_per_epoch, "epochs": epochs})
            cb.on_train_begin()
        it = iter(generator)
        for epoch in range(epochs):
            self.net.train()
            for cb in callbacks:
                cb.on_epoch_begin(epoch)
            loss_value = 0.0
            for step in range(steps_per_epoch):
                try:
                    x, y = next(it)
                except StopIteration:
                    it = iter(generator)
                    x, y = next(it)
                x, y = self._to_torch(x), self._to_torch(y)
                self._opt.zero_grad()
                loss = masked_bce(self.net(x), y)
                loss.backward()
                self._opt.step()
                loss_value = loss.item()
                for cb in callbacks:
                    cb.on_batch_end(step, logs={"loss": loss_value})
                if self.stop_training:
                    return
            logs = {"loss": loss_value}
            if validation_data is not None:
                self.net.eval()
                val_losses = []
                val_it = iter(validation_data)
                with torch.no_grad():
                    for _ in range(validation_steps or 1):
                        try:
                            x, y = next(val_it)
                        except StopIteration:
                            break
                        val_losses.append(masked_bce(self.net(self._to_torch(x)), self._to_torch(y)).item())
                if val_losses:
                    logs["val_loss"] = float(np.mean(val_losses))
            for cb in callbacks:
                cb.on_epoch_end(epoch, logs=logs)

    def predict(self, images, verbose=0):
        self.net.eval()
        with torch.no_grad():
            out = self.net(self._to_torch(np.asarray(images)))
        return out.permute(0, 2, 3, 1).cpu().numpy()

    # weight round-trip for Ais' continue-training path
    def get_weights(self):
        return [v.detach().cpu().numpy() for v in self.net.state_dict().values()]

    def set_weights(self, weights):
        sd = self.net.state_dict()
        self.net.load_state_dict({k: torch.as_tensor(w) for k, w in zip(sd.keys(), weights)})

    def save(self, path):
        torch.save({"input_shape": self.input_shape[1:], "state_dict": self.net.state_dict()}, path)

    @classmethod
    def load(cls, path):
        data = torch.load(path, map_location="cpu", weights_only=True)
        wrapper = cls(tuple(data["input_shape"]))
        wrapper.net.load_state_dict(data["state_dict"])
        wrapper.net.to(DEVICE).eval()
        return wrapper


def create(input_shape):
    return TorchWrapper(input_shape)


def load_model(path):
    return TorchWrapper.load(path)
```

