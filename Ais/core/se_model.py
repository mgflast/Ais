from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import load_model, clone_model, Model
from tensorflow.keras.layers import Input
from tensorflow.keras.backend import clear_session
import tensorflow as tf
import tifffile
import numpy as np
from itertools import count
import glob
import os
import sys
from contextlib import nullcontext
import Ais.core.config as cfg
import importlib
import threading
import json
from Ais.core.opengl_classes import Texture
from Ais.core.util import generate_thumbnail
from Ais.models.metrics import MaskedPrecision, MaskedRecall
from scipy.ndimage import binary_dilation, gaussian_filter
from skimage.transform import resize
import datetime
import time
import tarfile
import tempfile
import Ais.core.se_scnt as se_scnt
from Ais.core.normalization import global_stats, NORM_GLOBAL_MAD



class SEModelDataLoader:
    AUG_ROT90_XY = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
    AUG_FLIP_XY = [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]
    AUG_FLIP_Z = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
    P_AUG_BRIGHTNESS = 0.1
    P_AUG_CONTRAST = 0.1
    P_AUG_NOISE = 0.1
    P_AUG_SCALE = 0.0
    P_AUG_BLUR = 0.1
    P_AUG_GAMMA = 0.0

    def __init__(self, training_dataset_path, batch_size, validation_split, extra_augmentations=False):
        self.path = training_dataset_path
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.extra_augmentations = extra_augmentations
        self.apix = -1.0

        self.dataset = None
        self.box_shape = None
        self.box_depth = None
        self.z_jitter = 0            # extra stored Z-context for position jitter
        self.stored_depth = None     # slices per stored box (= box_depth + z_jitter)
        self.slab = False            # set by SEModel._train once the model type is known

        self.n_samples = 0
        self.idx_training_positive = []
        self.idx_training_all = []
        self.idx_validation_all = []

        self.load_data()

    def __len__(self):
        return self.n_samples

    def load_data(self):
        self.dataset = se_scnt.open_training_set(self.path)
        self.apix = self.dataset.apix
        self.normalization = getattr(self.dataset, "normalization", None)
        self.n_samples = self.dataset.n_samples
        self.box_shape = self.dataset.box_shape
        self.box_depth = self.dataset.box_depth
        self.z_jitter = getattr(self.dataset, "z_jitter", 0)
        self.stored_depth = getattr(self.dataset, "stored_depth", self.box_depth)

        idx_positive = self.dataset.positive_indices()
        if len(self.dataset.input_flavours) <= 1 or not hasattr(self.dataset, "flavour_coverage"):
            flavour_info = ""
        else:
            cov = self.dataset.flavour_coverage()
            flavour_info = f", flavour coverage (boxes): {', '.join(f'{k}={v}' for k, v in cov.items())}"
        if len(idx_positive) == 0:
            print(f'Training dataset at {self.path} contains no positive samples.')
        else:
            print(f'Loaded {self.n_samples} {self.box_shape}x{self.box_shape}x{self.box_depth} samples: {len(idx_positive)} positive, {self.n_samples - len(idx_positive)} negative{flavour_info}.')

        idx = np.arange(self.n_samples)
        np.random.shuffle(idx)
        n_validation = int(self.validation_split * self.n_samples)
        self.idx_training_all = idx[n_validation:]
        self.idx_validation_all = idx[:n_validation]
        self.idx_training_positive = [i for i in idx_positive if i in self.idx_training_all]

    def get_sample(self, index, training=True):
        x, y = self.dataset.get_sample(index, training=training)   # x (H,W,stored), y (H,W,1)
        if self.stored_depth == self.box_depth and not self.slab:
            return x, y
        return self._crop(x, y, training)

    def _crop(self, x, y, training):
        # Crop the model-depth window from the stored (jittered) box. For a slab model the crop
        # offset is random (so the annotated slice lands at a random Z-position, supervised via a
        # masked slab label); otherwise it's centred. The annotation sits at stored_depth//2.
        D, maxoff = self.box_depth, self.stored_depth - self.box_depth
        o = int(np.random.randint(0, maxoff + 1)) if (self.slab and training and maxoff > 0) else maxoff // 2
        x = x[:, :, o:o + D]
        if self.slab:
            r = self.stored_depth // 2 - o                          # annotated slice's index in the crop
            yl = np.full((y.shape[0], y.shape[1], D, 1), 2.0, dtype=np.float32)   # ignore everywhere
            yl[:, :, r, 0] = y[:, :, 0]                              # real label only on the annotated slice
            y = yl
        return x, y

    @staticmethod
    def _augment_brightness(x, y):
        x += np.random.uniform(-0.1, 0.1)
        return x, y

    @staticmethod
    def _augment_contrast(x, y):
        x *= np.random.uniform(0.9, 1.1)
        return x, y

    @staticmethod
    def _augment_gaussian_noise( x, y):
        noise = np.random.normal(0, 0.2, size=x.shape)
        x += noise
        return x, y

    @staticmethod
    def _augment_scale(x, y):
        factor = np.random.uniform(0.9, 1.1)
        if np.isclose(factor, 1.0):
            return x, y

        H, W, D = x.shape

        new_shape = np.round((x.shape[0] * factor, x.shape[1] * factor, x.shape[2])).astype(int)
        zoomed_img = resize(x, new_shape, order=1, anti_aliasing=True)
        zoomed_label = (resize(y, new_shape, order=0, anti_aliasing=False) > 0).astype(np.float32)

        zh, zw, _ = zoomed_img.shape

        if factor < 1.0:
            pad_h = H - zh
            pad_w = W - zw

            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left

            x = np.pad(zoomed_img, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='reflect')
            y = np.pad(zoomed_label, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant', constant_values=2)
        else:
            start_h = (zh - H) // 2
            start_w = (zw - W) // 2
            x = zoomed_img[start_h:start_h + H, start_w:start_w + W, :]
            y = zoomed_label[start_h:start_h + H, start_w:start_w + W, :]

        return x, y

    @staticmethod
    def _augment_blur(x, y):
        sigma = np.random.uniform(0.1, 1.1)
        x = gaussian_filter(x, sigma)
        return x, y

    @staticmethod
    def _augment_gamma(x, y):
        gamma = np.random.uniform(0.8, 1.2)
        x = np.sign(x) * (np.abs(x) ** gamma)
        return x, y

    @staticmethod
    def _extra_augmentations(x, y):
        if np.random.uniform() < SEModelDataLoader.P_AUG_BRIGHTNESS:
            x, y = SEModelDataLoader._augment_brightness(x, y)
        if np.random.uniform() < SEModelDataLoader.P_AUG_CONTRAST:
            x, y = SEModelDataLoader._augment_contrast(x, y)
        if np.random.uniform() < SEModelDataLoader.P_AUG_NOISE:
            x, y = SEModelDataLoader._augment_gaussian_noise(x, y)
        if np.random.uniform() < SEModelDataLoader.P_AUG_SCALE:
            x, y = SEModelDataLoader._augment_scale(x, y)
        if np.random.uniform() < SEModelDataLoader.P_AUG_BLUR:
            x, y = SEModelDataLoader._augment_blur(x, y)
        if np.random.uniform() < SEModelDataLoader.P_AUG_GAMMA:
            x, y = SEModelDataLoader._augment_gamma(x, y)
        return x, y

    def augment(self, x, y):
        _ = np.random.randint(16 if self.box_depth > 1 else 8)  # is self.box_depth > 1, we're training a 2.5 or 3D Net, with shape (x, y, z).
        x = np.rot90(x, k=SEModelDataLoader.AUG_ROT90_XY[_], axes=(0, 1))
        y = np.rot90(y, k=SEModelDataLoader.AUG_ROT90_XY[_], axes=(0, 1))
        if SEModelDataLoader.AUG_FLIP_XY[_]:
            x = np.flip(x, axis=0)
            y = np.flip(y, axis=0)
        if SEModelDataLoader.AUG_FLIP_Z[_] and self.box_depth > 1:
            x = np.flip(x, axis=2)             # x: (H,W,D) -> flip Z
            if y.ndim == 4:                    # slab label (H,W,D,1) -> flip Z too (keeps label aligned)
                y = np.flip(y, axis=2)

        if self.extra_augmentations:
            x, y = SEModelDataLoader._extra_augmentations(x, y)
        return x, y

    def preprocess(self, x, y):
        if self.normalization == se_scnt.NORM_GLOBAL_MAD:   # global boxes are pre-normalized
            return x, y
        x -= np.mean(x)
        x /= np.std(x) + 1e-7
        return x, y

    def training_generator(self):
        while True:
            np.random.shuffle(self.idx_training_all)
            np.random.shuffle(self.idx_training_positive)
            for j in range(len(self.idx_training_all)):
                if j % self.batch_size == 0:  # at least one labelled sample per batch
                    index = self.idx_training_positive[(j // self.batch_size) % len(self.idx_training_positive)]
                else:
                    index = self.idx_training_all[j]

                x, y = self.get_sample(index, training=True)
                x, y = self.preprocess(x, y)
                x, y = self.augment(x, y)

                yield x, y

    def validation_generator(self):
        while True:
            for index in self.idx_validation_all:
                x, y = self.get_sample(index, training=False)
                x, y = self.preprocess(x, y)
                yield x, y

    def _y_signature(self):
        if self.slab:   # slab label matches the model's (Y, X, Z, 1) output
            return tf.TensorSpec(shape=(self.box_shape, self.box_shape, self.box_depth, 1), dtype=tf.float32)
        return tf.TensorSpec(shape=(self.box_shape, self.box_shape, 1), dtype=tf.float32)

    def as_generator(self, validation=False):
        if validation:
            while self.batch_size > len(self.idx_validation_all):
                self.batch_size = self.batch_size // 2
            n_steps = len(self.idx_validation_all) // self.batch_size
            dataset = tf.data.Dataset.from_generator(self.validation_generator, output_signature=(tf.TensorSpec(shape=(self.box_shape, self.box_shape, self.box_depth), dtype=tf.float32), self._y_signature())).batch(batch_size=self.batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
        else:
            while self.batch_size > len(self.idx_training_all):
                self.batch_size = self.batch_size // 2

            n_steps = len(self.idx_training_all) // self.batch_size
            dataset = tf.data.Dataset.from_generator(self.training_generator, output_signature=(tf.TensorSpec(shape=(self.box_shape, self.box_shape, self.box_depth), dtype=tf.float32), self._y_signature())).batch(batch_size=self.batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

        return dataset, n_steps


class SEModel:
    idgen = count(0)
    AVAILABLE_MODELS = []
    MODELS = dict()
    MODELS_LOADED = False
    DEFAULT_COLOURS = [(66 / 255, 214 / 255, 164 / 255),
                       (255 / 255, 243 / 255, 0 / 255),
                       (255 / 255, 104 / 255, 0 / 255),
                       (255 / 255, 13 / 255, 0 / 255),
                       (174 / 255, 0 / 255, 255 / 255),
                       (21 / 255, 0 / 255, 255 / 255),
                       (0 / 255, 136 / 255, 266 / 255),
                       (0 / 255, 247 / 255, 255 / 255),
                       (0 / 255, 255 / 255, 0 / 255)]
    DEFAULT_MODEL_ENUM = 1

    def __init__(self, no_glfw=False):
        if not SEModel.MODELS_LOADED:
            SEModel.load_models()

        uid_counter = next(SEModel.idgen)
        self.uid = int(datetime.datetime.now().strftime('%Y%m%d%H%M%S%f') + "000") + uid_counter
        self.title = "Unnamed model"
        self.colour = SEModel.DEFAULT_COLOURS[(uid_counter) % len(SEModel.DEFAULT_COLOURS)]
        self.apix = -1.0
        self.compiled = False
        self.compilation_mode = None
        self.inference_model = None
        self.box_size = -1
        self.model_depth = 1
        self.model = None
        self.model_enum = SEModel.DEFAULT_MODEL_ENUM
        self.epochs = 50
        self.batch_size = 32
        self.train_data_path = "(path to training dataset)"
        self.active = True
        self.export = True
        self.blend = False
        self.show = True
        self.alpha = 0.75
        self.threshold = 0.5
        self.overlap = 0.2
        self.active_tab = 0
        self.background_process_train = None
        self.background_process_apply = None
        self.n_parameters = 0
        self.n_copies = 10
        self.excess_negative = -100
        self.info = ""
        self.info_short = ""
        self.loss = 0.0
        self.normalization = None  # None = legacy per-box/per-slice; else global
        self.z_jitter = 0          # slab models: trained Z-position jitter (= reliable output half-range*2)
        self.tta = 1               # per-model test-time augmentation multiplicity (1-8)
        self.data = None
        if not no_glfw:
            self.texture = Texture(format="r32f")
            self.texture.set_linear_mipmap_interpolation()
        self.bcprms = dict()  # backward compatibility params dict.
        self.emit = False
        self.absorb = False
        self.interactions = list()  # list of ModelInteraction objects.

    def delete(self):
        for interaction in self.interactions:
            ModelInteraction.all.remove(interaction)

    def save(self, file_path, validation_slice=None):
        with tempfile.TemporaryDirectory() as temp_dir:
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            model_path = os.path.join(temp_dir, base_name + '_weights.h5')
            metadata_path = os.path.join(temp_dir, base_name + "_metadata.json")
            slice_path = None
            thumbnail_path = None

            self.model.save(model_path)
            if validation_slice is not None:
                slice_path = os.path.join(temp_dir, base_name + "_slice.tiff")
                tifffile.imwrite(slice_path, validation_slice)

                thumbnail_path = os.path.join(temp_dir, base_name + "_preview.png")
                segmentation = self.apply_to_slice(validation_slice, 1.0)
                thumbnail = generate_thumbnail(validation_slice, segmentation, self.colour)
                thumbnail.save(thumbnail_path)

            # Save metadata
            metadata = {
                'title': self.title,
                'colour': self.colour,
                'apix': self.apix,
                'compiled': self.compiled,
                'box_size': self.box_size,
                'model_depth': self.model_depth,
                'model_enum': self.model_enum,
                'epochs': self.epochs,
                'batch_size': self.batch_size,
                'active': self.active,
                'blend': self.blend,
                'show': self.show,
                'alpha': self.alpha,
                'threshold': self.threshold,
                'overlap': self.overlap,
                'active_tab': self.active_tab,
                'n_parameters': self.n_parameters,
                'n_copies': self.n_copies,
                'info': self.info,
                'info_short': self.info_short,
                'excess_negative': self.excess_negative,
                'emit': self.emit,
                'absorb': self.absorb,
                'loss': self.loss,
                'normalization': self.normalization,
                'z_jitter': self.z_jitter,
                'tta': self.tta
            }
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)

            with tarfile.open(file_path, 'w') as archive:
                archive.add(model_path, arcname=os.path.basename(model_path))
                archive.add(metadata_path, arcname=os.path.basename(metadata_path))
                if slice_path:
                    archive.add(slice_path, arcname=os.path.basename(slice_path))
                    archive.add(thumbnail_path, arcname=os.path.basename(thumbnail_path))

    @staticmethod
    def load_metadata(file_path):
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                with tarfile.open(file_path, 'r') as archive:
                    archive.extractall(path=temp_dir)

                metadata_file = glob.glob(os.path.join(temp_dir, "*_metadata.json"))[0]

                # Load metadata
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)


                return metadata

        except Exception as e:
            print("Error loading model - see details below\n", e)

    def load(self, file_path, compile=False):
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                with tarfile.open(file_path, 'r') as archive:
                    archive.extractall(path=temp_dir)

                model_file = glob.glob(os.path.join(temp_dir, "*_weights.h5"))[0]
                metadata_file = glob.glob(os.path.join(temp_dir, "*_metadata.json"))[0]

                # Load the Keras model
                self.model = load_model(model_file, compile=compile)
                self.toggle_inference()
                # Load metadata
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)

                self.title = metadata['title']
                self.colour = metadata['colour']
                self.apix = metadata['apix']
                self.compiled = metadata['compiled']
                self.box_size = metadata['box_size']
                self.model_depth = metadata.get('model_depth', 1)
                self.model_enum = metadata['model_enum']
                self.epochs = metadata['epochs']
                self.batch_size = metadata['batch_size']
                self.active = metadata['active']
                self.blend = metadata['blend']
                self.show = metadata['show']
                self.alpha = metadata['alpha']
                self.threshold = metadata['threshold']
                self.overlap = metadata['overlap']
                self.active_tab = metadata['active_tab']
                self.n_parameters = metadata['n_parameters']
                self.n_copies = metadata['n_copies']
                self.info = metadata['info']
                self.info_short = metadata['info_short']
                self.excess_negative = metadata['excess_negative']
                self.emit = metadata['emit']
                self.absorb = metadata['absorb']
                self.loss = metadata['loss']
                self.normalization = metadata.get('normalization', None)  # absent => legacy
                self.z_jitter = metadata.get('z_jitter', 0)
                self.tta = metadata.get('tta', 1)

                # a CLI-trained model keeps a default model colour; if its title matches a feature
                # in the active library, adopt that feature's colour (don't touch custom colours).
                if any(all(abs(self.colour[i] - d[i]) < 1e-3 for i in range(3)) for d in SEModel.DEFAULT_COLOURS):
                    for f in cfg.feature_library:
                        if f.title == self.title:
                            self.colour = tuple(f.colour)
                            break

        except Exception as e:
            print("Error loading model - see details below\n", e)

    def train(self, rate=None, external_callbacks=None, extra_augmentations=False, strategy=None):
        if self.train_data_path:
            # Model build/compile (incl. the inference->training rebuild) is deferred to _train
            # so it runs inside strategy.scope() in the worker thread that calls fit() - the only
            # place a MirroredStrategy actually distributes the fit.
            process = BackgroundProcess(self._train, (rate, external_callbacks, extra_augmentations, strategy), name=f"{self.title} training")
            self.background_process_train = process
            self.inference_model = None
            process.start()

    def _train(self, rate=None, external_callbacks=None, extra_augmentations=False, strategy=None, process=None):
        try:
            start_time = time.time()
            validation_split = 0.0 if "VALIDATION_SPLIT" not in self.bcprms else self.bcprms["VALIDATION_SPLIT"]

            loader = SEModelDataLoader(self.train_data_path, self.batch_size, validation_split, extra_augmentations=extra_augmentations)
            self.model_depth = loader.box_depth
            self.apix = loader.apix
            self.normalization = loader.normalization

            # Continuing from an inference-mode model: pull out its weights and reset the graph
            # *before* entering strategy.scope(). clear_session() resets the default graph and
            # empties TF's per-graph distribution-strategy stack, so doing it inside the scope
            # crashes on scope exit (IndexError: pop from empty list). The rebuild happens below.
            restore_weights = None
            if self.compilation_mode == 'inference':
                restore_weights = self.model.get_weights()
                del self.model
                self.model = None
                self.compiled = False
                clear_session()

            # Build/compile and fit inside the scope so MirroredStrategy variables are created in
            # this worker thread. nullcontext() = single-GPU / GUI (no distribution).
            with (strategy.scope() if strategy is not None else nullcontext()):
                if not self.compiled:
                    self.compile(loader.box_shape, loader.box_depth)
                if restore_weights is not None:
                    self.model.set_weights(restore_weights)

                # model type now known: rank-5 output => Z-slab model. It needs the loader to
                # random-crop + build masked slab labels (Z-position jitter); carry the jitter onto
                # the model so inference knows the reliable output range.
                loader.slab = (len(self.model.output_shape) == 5)
                self.z_jitter = loader.z_jitter
                if loader.slab and loader.z_jitter == 0:
                    print("Warning: 3D slab model trained on data with no Z-jitter (z_jitter=0) - only "
                          "the central slice will be well-supervised. Re-extract at a slab --box-depth.")

                learning_rate = rate if rate is not None else cfg.settings["LEARNING_RATE"]
                self.model.optimizer.learning_rate.assign(learning_rate)

                training_generator, training_steps = loader.as_generator(validation=False)
                validation_generator, validation_steps = None, None
                if validation_split != 0.0:
                    validation_generator, validation_steps = loader.as_generator(validation=True)

                if training_steps == 0:
                    cfg.set_error(ValueError("No training samples available. Aborting."), "Could not train model - see details below.")

                callbacks = [TrainingProgressCallback(process, training_steps * self.n_copies, self.batch_size, self), StopTrainingCallback(process.stop_request)]
                if not external_callbacks is None:
                    callbacks += external_callbacks

                self.model.fit(training_generator, steps_per_epoch=training_steps * self.n_copies, validation_data=validation_generator, validation_steps=validation_steps, epochs=self.epochs, validation_freq=1, callbacks=callbacks)
            process.set_progress(1.0)
            print(f"{self.title} " + self.info + f" {time.time() - start_time:.1f} seconds of training.")
        except Exception as e:
            cfg.set_error(e, "Could not train model - see details below.")
            process.stop()

    def reset_textures(self):
        pass

    def compile(self, box_size, box_depth=1):
        model_module_name = SEModel.AVAILABLE_MODELS[self.model_enum]
        self.model = SEModel.MODELS[model_module_name]((box_size, box_size, box_depth))
        # Re-compile with the arch's own loss/optimizer plus streaming precision/recall,
        # so they are printed per epoch during training (like easymode). Masked metrics
        # respect the ignore label (2) and treat label==1 as foreground.
        self.model.compile(optimizer=self.model.optimizer, loss=self.model.loss,
                           metrics=[MaskedPrecision(), MaskedRecall()])
        self.compiled = True
        self.compilation_mode = 'training'
        self.box_size = box_size
        self.n_parameters = self.model.count_params()
        self.update_info()

    def toggle_inference(self):
        config = self.model.get_config()
        weights = self.model.get_weights()

        del self.model
        clear_session()

        input_shape = list(config["layers"][0]["config"]["batch_input_shape"])
        input_shape[0] = 1
        input_shape[1] = None
        input_shape[2] = None

        self.model = None
        config["layers"][0]["config"]["batch_input_shape"] = tuple(input_shape)
        self.model = Model.from_config(config)
        self.model.set_weights(weights)

        self.compilation_mode = 'inference'

    def toggle_training(self):
        weights = self.model.get_weights()

        del self.model
        clear_session()

        self.model = None
        self.compile(self.box_size, self.model_depth)
        self.model.set_weights(weights)

        self.compilation_mode = 'training'

    def update_info(self):
        validation_split_tag = "" if ("VALIDATION_SPLIT" not in self.bcprms or self.bcprms["VALIDATION_SPLIT"] == 0.0) else f"|{int(self.bcprms['VALIDATION_SPLIT']*100.0)}%"
        if self.compilation_mode == 'training':
            self.info = SEModel.AVAILABLE_MODELS[self.model_enum] + f" ({self.n_parameters // 1e6:.1f} Mp, {self.box_size}-{self.model_depth}, {self.apix:.1f}, {self.loss:.4f}{validation_split_tag})"
            self.info_short = "(" + SEModel.AVAILABLE_MODELS[self.model_enum] + f", {self.box_size}-{self.model_depth}, {self.apix:.1f}, {self.loss:.4f}{validation_split_tag})"
        elif self.compilation_mode == 'inference':
            self.info = SEModel.AVAILABLE_MODELS[self.model_enum] + f" ({self.n_parameters // 1e6:.1f} Mp, {self.box_size}-{self.model_depth}, {self.apix:.1f}, {self.loss:.4f}{validation_split_tag})"
            self.info_short = "(" + SEModel.AVAILABLE_MODELS[self.model_enum] + f", {self.box_size}-x{self.model_depth}, {self.apix:.1f}, {self.loss:.4f}{validation_split_tag})"

    def get_model_title(self):
        return SEModel.AVAILABLE_MODELS[self.model_enum]

    def set_slice(self, slice_data, slice_pixel_size, roi, original_size, norm_stats=None):
        try:
            self.data = np.zeros(original_size)
            if not self.compiled:
                return False
            if not self.active:
                return False
            rx, ry = roi
            self.data[rx[0]:rx[1], ry[0]:ry[1]] = self.apply_to_slice(slice_data[:, rx[0]:rx[1], ry[0]:ry[1]], slice_pixel_size, norm_stats=norm_stats)
            return True
        except Exception as e:
            print(e)
            return False

    def update_texture(self):
        if not cfg.glfw_initialized:
            return
        if not self.compiled or not self.active:
            return
        self.texture.update(self.data)

    def slice_to_boxes(self, image, pixel_size, as_array=True):
        w, h, d = image.shape
        self.overlap = min([0.9, self.overlap])
        pad_w = self.box_size - (w % self.box_size)
        pad_h = self.box_size - (h % self.box_size)
        # tile
        stride = int(self.box_size * (1.0 - self.overlap))
        boxes = list()
        image = np.pad(image, ((0, pad_w), (0, pad_h), (0, 0)), mode='reflect')
        global_norm = cfg.settings["NORMALIZATION"] == "global"
        for x in range(0, w + pad_w - self.box_size + 1, stride):
            for y in range(0, h + pad_h - self.box_size + 1, stride):
                box = image[x:x + self.box_size, y:y + self.box_size, :]
                if not global_norm:   # legacy: per-box
                    mu = np.mean(box, axis=(0, 1, 2), keepdims=True)
                    std = np.std(box, axis=(0, 1, 2), keepdims=True) + 1e-7
                    box = (box - mu) / std
                boxes.append(box)
        if as_array:
            boxes = np.array(boxes)
        return boxes, (w, h), (pad_w, pad_h), stride

    def boxes_to_slice(self, boxes, size, original_pixel_size, padding, stride):
        pad_w, pad_h = padding
        w, h = size
        out_image = np.zeros((w + pad_w, h + pad_h))
        count = np.zeros((w + pad_w, h + pad_h), dtype=int)
        # 240809: apply a mask to all the segmented boxes so that only center (best) bit us used
        box_size = boxes[0].shape[0]
        mask = np.ones((box_size, box_size), dtype=int)
        overlap = 1 - (stride / box_size)
        margin = int(overlap * box_size / 2)
        if margin > 0:
            mask[-margin:, :] = 0.0
            mask[:margin, :] = 0.0
            mask[:, -margin:] = 0.0
            mask[:, :margin] = 0.0

        if cfg.settings["OVERLAP_MODE"] == 0:
            mask[:, :] = 1
        i = 0
        for x in range(0, w + pad_w - self.box_size + 1, stride):
            for y in range(0, h + pad_h - self.box_size + 1, stride):
                out_image[x:x + self.box_size, y:y + self.box_size] += boxes[i] * mask
                count[x:x + self.box_size, y:y + self.box_size] += mask
                i += 1
        c_mask = count == 0
        count[c_mask] = 1
        out_image[c_mask] = 0
        out_image = out_image / count
        out_image = out_image[:w, :h]
        out_image = out_image[:w, :h]
        return out_image

    def is_3d(self):
        # a slab model has a rank-5 input (Y, X, Z, C); these always tile at inference
        return self.model is not None and len(self.model.input_shape) == 5

    def _run(self, x):
        """Run the compiled model on x=(B, Y, X, D) and return (B, Y, X). A 3D model wants a
        trailing channel (B, Y, X, D, 1); its loaded inference graph fixes batch=1 (Conv3DTranspose),
        so run it one sample at a time, and reduce a slab output to its central Z slice (the GUI
        only ever shows the current slice)."""
        if len(self.model.input_shape) == 5:
            outs = []
            for b in range(x.shape[0]):
                o = self.model.predict(x[b][np.newaxis, ..., np.newaxis], verbose=0)   # (1,Y,X,D,1) or (1,Y,X,1)
                if o.ndim == 5:
                    o = o[:, :, :, o.shape[3] // 2, :]     # slab -> central Z slice
                outs.append(o[0])                          # (Y, X, 1)
            return np.squeeze(np.stack(outs, axis=0), axis=-1)   # (B, Y, X)
        out = self.model.predict(x, verbose=0)             # (B, Y, X, 1)
        return np.squeeze(out, axis=-1)                    # (B, Y, X)

    def apply_to_slice(self, image, pixel_size, norm_stats=None):
        # as of 251125: we always expect image to be a 3D numpy array; it is Z, Y, X when coming in to this function, but note that the models expects Z last.

        is_3d = (len(self.model.input_shape) == 5)   # 3D slab models always tile (a full slice is too big in 3D)
        if self.compilation_mode == 'training' and self.background_process_train is None and (cfg.settings["TILED_MODE"] == 0 or is_3d):
            self.toggle_inference()

        start_time = time.time()

        if image.ndim != 3:
            cfg.set_error(ValueError("Input image must be a 3D numpy array."), "SegmentationEditor model application error:")

        orig_y, orig_x = image.shape[1], image.shape[2]
        tomo_rescaled = False
        tomo_rescale_factor = 1.0
        if cfg.settings["INFERENCE_ALLOW_RESCALING"] and pixel_size != -1 and self.apix != -1:
            tomo_rescale_factor = 10.0 * pixel_size / self.apix  # pixel_size is in nm...
            if abs(tomo_rescale_factor - 1.0) > 0.05:
                print(f'Scaling tomogram by a factor {tomo_rescale_factor}')
                new_shape = np.round([image.shape[0], orig_y * tomo_rescale_factor, orig_x * tomo_rescale_factor]).astype(int)
                image = resize(image, new_shape, order=1, anti_aliasing=True)
                tomo_rescaled = True

        image = np.transpose(image, (1, 2, 0))
        j, k, image_depth = image.shape

        global_norm = cfg.settings["NORMALIZATION"] == "global"
        if global_norm:   # normalize whole slab once; caller passes the whole-volume stat
            center, scale = norm_stats if norm_stats is not None else global_stats(np.transpose(image, (2, 0, 1)))
            image = (image - center) / scale

        if self.compilation_mode == 'inference' and cfg.settings["TILED_MODE"]==0 and not is_3d:
            if not global_norm:   # legacy: whole-slice
                image -= np.mean(image)
                image /= np.std(image) + 1e-7

            pad_h_min = (32 - (image.shape[0] % 32)) % 32
            pad_w_min = (32 - (image.shape[1] % 32)) % 32
            pad_h = pad_h_min + 64
            pad_w = pad_w_min + 64

            image = np.pad(image, ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2), (0, 0)), mode='reflect')
            segmentation = np.zeros(image.shape[:2], dtype=np.float32)

            _rot = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
            _flip = [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]
            _flip_z = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]

            _tta = self.tta if image_depth > 1 else min(self.tta, 8)
            for i in range(_tta):
                # rotate and flip
                tta_img = np.rot90(image, k=_rot[i], axes=(0, 1))
                if _flip[i]:
                    tta_img = np.flip(tta_img, axis=0)
                if _flip_z[i]:
                    tta_img = np.flip(tta_img, axis=2)

                tta_img_segmented = self._run(tta_img[np.newaxis, ...])[0]

                # undo rotation and flip
                if _flip[i]:
                    tta_img_segmented = np.flip(tta_img_segmented, axis=0)
                tta_img_segmented = np.rot90(tta_img_segmented, k=-_rot[i], axes=(0, 1))

                segmentation += tta_img_segmented
            segmentation = segmentation[pad_h//2:pad_h//2+j, pad_w//2:pad_w//2+k] / _tta
            print(self.info + f" cost for {segmentation.shape[0]}x{segmentation.shape[1]} slice: {time.time() - start_time:.3f} s (multiplicity {self.tta}).")
        else:  # tiled inference (also used for every 3D model); TTA wraps the tiled pass
            _rot = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
            _flip = [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]
            _flip_z = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
            _tta = self.tta if image_depth > 1 else min(self.tta, 8)
            segmentation = None
            for i in range(_tta):
                tta_img = np.rot90(image, k=_rot[i], axes=(0, 1))
                if _flip[i]:
                    tta_img = np.flip(tta_img, axis=0)
                if _flip_z[i]:
                    tta_img = np.flip(tta_img, axis=2)
                boxes, image_size, padding, stride = self.slice_to_boxes(tta_img, pixel_size)
                seg_i = self.boxes_to_slice(self._run(boxes), image_size, pixel_size, padding, stride)
                if _flip[i]:
                    seg_i = np.flip(seg_i, axis=0)
                seg_i = np.rot90(seg_i, k=-_rot[i], axes=(0, 1))
                segmentation = seg_i if segmentation is None else segmentation + seg_i
            segmentation = segmentation / _tta
            print(self.info + f" cost for {segmentation.shape[0]}x{segmentation.shape[1]} slice (tiled, tta {_tta}): {time.time()-start_time:.3f} s.")

        if cfg.settings["INFERENCE_ALLOW_RESCALING"] and tomo_rescaled:
            seg_rescaled = resize(segmentation, [orig_y, orig_x], order=1, anti_aliasing=True)

            # Make sure the size matches exactly the original in-plane size
            sy, sx = seg_rescaled.shape

            # Crop or pad in Y
            if sy > orig_y:
                seg_rescaled = seg_rescaled[:orig_y, :]
            elif sy < orig_y:
                pad_y = orig_y - sy
                seg_rescaled = np.pad(seg_rescaled, ((0, pad_y), (0, 0)), mode="edge")

            # Crop or pad in X
            sy, sx = seg_rescaled.shape
            if sx > orig_x:
                seg_rescaled = seg_rescaled[:, :orig_x]
            elif sx < orig_x:
                pad_x = orig_x - sx
                seg_rescaled = np.pad(seg_rescaled, ((0, 0), (0, pad_x)), mode="edge")

            segmentation = seg_rescaled


        if cfg.settings["TRIM_EDGES"] == 1:
            margin = 16
            segmentation[:margin, :] = 0
            segmentation[-margin:, :] = 0
            segmentation[:, :margin] = 0
            segmentation[:, -margin:] = 0

        return segmentation

    @staticmethod
    def load_models():
        model_files = glob.glob(os.path.join(cfg.root, "models", "*.py"))
        model_files_user = glob.glob(os.path.join(os.path.dirname(cfg.settings_path), "models", "*.py"))
        model_files = sorted(model_files + model_files_user, key=lambda x: os.path.basename(x))
        for file in model_files:
            module_name = os.path.basename(file)[:-3]
            try:
                # if in default library:
                if os.path.dirname(file) == os.path.join(cfg.root, "models"):
                    mod = importlib.import_module(("Ais." if not cfg.frozen else "")+"models."+module_name)
                else:
                    spec = importlib.util.spec_from_file_location(module_name, file)
                    mod = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = mod
                    spec.loader.exec_module(mod)
                if mod.include:
                    SEModel.MODELS[mod.title] = mod.create
            except Exception as e:
                cfg.set_error(e, "Could not load SegmentationEditor model at path: "+file)
        SEModel.MODELS_LOADED = True
        SEModel.AVAILABLE_MODELS = list(SEModel.MODELS.keys())
        if 'VGGNet M' in SEModel.AVAILABLE_MODELS:
            SEModel.DEFAULT_MODEL_ENUM = SEModel.AVAILABLE_MODELS.index('VGGNet M')

    def __eq__(self, other):
        if isinstance(other, SEModel):
            return self.uid == other.uid
        return False


class ModelInteraction:
    idgen = count(0)
    TYPES = ["Colocalize", "Avoid"]

    all = list()

    def __init__(self, parent, partner):
        self.uid = int(datetime.datetime.now().strftime('%Y%m%d%H%M%S%f') + "000") + next(ModelInteraction.idgen)
        self.parent = parent
        self.partner = partner
        self.type = 0
        self.radius = 10.0  # nanometer
        self.threshold = 0.5
        self.kernel = np.zeros((1, 1))
        self.kernel_info = "none"
        ModelInteraction.all.append(self)

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return self.uid == other.uid
        return False

    def get_kernel(self, pixel_size):
        info_str = f"{self.radius}_{pixel_size}"
        if self.kernel_info == info_str:  # check if the previously generated kernel was the same (i.e. same radius, pixel_size). If so, return it, else, compute the new kernel and return that.
            return self.kernel
        else:
            radius_pixels = int(self.radius // pixel_size)
            self.kernel = np.zeros((radius_pixels * 2 + 1, radius_pixels * 2 + 1), dtype=np.float32)
            r2 = radius_pixels**2
            for x in range(0, 2*radius_pixels+1):
                for y in range(0, 2*radius_pixels+1):
                    self.kernel[x, y] = 1.0 if ((x-radius_pixels)**2 + (y-radius_pixels)**2) < r2 else 0.0
            self.kernel_info = f"{self.radius}_{pixel_size}"
            return self.kernel

    def apply(self, pixel_size):
        print(f"Applying model interaction {self.uid}")
        if self.parent.active:
            self.parent.data = self.apply_to_images(pixel_size, self.partner.data, self.parent.data)

    def apply_to_images(self, pixel_size, partner_image, parent_image):
        int_mask = False
        if partner_image.dtype == np.uint8:
            partner_mask = np.where(partner_image > self.threshold * 255, 1, 0)
            int_mask = True
        else:
            partner_mask = np.where(partner_image > self.threshold, 1, 0)
        kernel = self.get_kernel(pixel_size)
        if self.type == 0:
            mask = binary_dilation(partner_mask, structure=kernel).astype(np.float32 if not int_mask else np.uint8)
            parent_image = parent_image * mask  # this might be sped up by in place multiplication; [] *= []
        elif self.type == 1:
            mask = 1.0 - binary_dilation(partner_mask, structure=kernel).astype(np.float32 if not int_mask else np.uint8)
            parent_image = parent_image * mask
        return parent_image

    def as_dict(self):
        mdict = dict()
        mdict['parent_title'] = self.parent.title
        mdict['partner_title'] = self.partner.title
        mdict['type'] = self.type
        mdict['radius'] = self.radius
        mdict['threshold'] = self.threshold
        return mdict

    @staticmethod
    def from_dict(mdict):
        parent_title = mdict['parent_title']
        partner_title = mdict['partner_title']
        partner_model = None
        parent_model = None
        for m in cfg.se_models:
            if m.title == parent_title:
                parent_model = m
            elif m.title == partner_title:
                partner_model = m
        if partner_model is None or parent_model is None:
            return
        interaction = ModelInteraction(parent_model, partner_model)
        interaction.type = mdict['type']
        interaction.radius = mdict['radius']
        interaction.threshold = mdict['threshold']
        parent_model.interactions.append(interaction)


class TrainingProgressCallback(Callback):
    def __init__(self, process, n_samples, batch_size, model):
        super().__init__()
        self.process = process
        self.batch_size = batch_size
        self.n_samples = n_samples  # kept for compatibility / logging if you want
        self.se_model = model
        self.total_batches = None
        self.seen_batches = 0

    def on_train_begin(self, logs=None):
        # Keras fills self.params before this is called
        steps = self.params.get('steps', None)      # steps_per_epoch
        epochs = self.params.get('epochs', 1)
        if steps is None:
            # fallback to old behaviour if steps missing
            steps = max(1, self.n_samples // self.batch_size)
        self.total_batches = steps * epochs
        self.seen_batches = 0

    def on_epoch_begin(self, epoch, logs=None):
        pass  # nothing needed here now

    def on_epoch_end(self, epoch, logs=None):
        if "VALIDATION_LOSS" in self.se_model.bcprms and not self.se_model.bcprms["VALIDATION_LOSS"] == 0.0:
            return
        if logs is not None:
            val_loss = logs.get('val_loss')
            if val_loss is not None:
                self.se_model.loss = val_loss
                self.se_model.update_info()

    def on_batch_end(self, batch, logs=None):
        self.seen_batches += 1
        if self.total_batches is None:
            progress = 0.0
        else:
            progress = min(1.0, self.seen_batches / self.total_batches)

        self.process.set_progress(progress)

        if "VALIDATION_LOSS" not in self.se_model.bcprms or self.se_model.bcprms["VALIDATION_LOSS"] == 0.0:
            if logs is not None and 'loss' in logs:
                self.se_model.loss = logs['loss']
        self.se_model.update_info()


class StopTrainingCallback(Callback):
    def __init__(self, stop_request):
        self.params = dict()
        self.params['epochs'] = 1
        super().__init__()
        self.stop_request = stop_request

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        if self.stop_request.is_set():
            self.model.stop_training = True


class BackgroundProcess:
    idgen = count(0)

    def __init__(self, function, args, name=None):
        self.uid = next(BackgroundProcess.idgen)
        self.function = function
        self.args = args
        self.name = name
        self.thread = None
        self.progress = 0.0
        self.stop_request = threading.Event()

    def start(self):
        _name = f"BackgroundProcess {self.uid} - "+(self.name if self.name is not None else "")
        self.thread = threading.Thread(daemon=True, target=self._run, name=_name)
        self.thread.start()

    def _run(self):
        self.function(*self.args, self)

    def set_progress(self, progress):
        self.progress = progress

    def stop(self):
        self.stop_request.set()
        self.progress = 1.0

    def __str__(self):
        return f"BackgroundProcess {self.uid} with function {self.function} and args {self.args}"

