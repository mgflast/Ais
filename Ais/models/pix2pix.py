from tensorflow.keras.layers import Input, Concatenate, UpSampling2D, Conv2D, BatchNormalization, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import CallbackList
import tensorflow as tf
from .losses import masked_bce

# adapted from source: https://github.com/eriklindernoren/Keras-GAN
title = "Pix2pix"
include = True


def create(input_shape):
    if input_shape[0] % 32:
        raise ValueError(f"Pix2pix needs a box size that is a multiple of 32 (got {input_shape[0]}).")
    return Pix2Pix(input_shape)


def load_model(path):
    return Pix2Pix(generator=tf.keras.models.load_model(path, compile=False))


class Pix2Pix:
    # Not a tf.keras.Model: SEModel's isinstance(Model) guards skip the functional-only rebuilds,
    # and this class provides the rest of the Model surface that SEModel uses.
    LAMBDA = 100.0
    gf = 64
    df = 64

    def __init__(self, input_shape=None, generator=None):
        channels = generator.input_shape[-1] if generator is not None else input_shape[-1]
        self.generator = generator if generator is not None else self.build_generator(channels)
        self.discriminator = self.build_discriminator(channels)
        self.optimizer = Adam(2e-4, 0.5)
        self.d_optimizer = Adam(2e-4, 0.5)
        self.loss = masked_bce
        self.metrics = []
        self.stop_training = False

    def build_generator(self, channels):
        def conv2d(layer_input, filters, f_size=4, bn=True):
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4):
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            u = BatchNormalization(momentum=0.8)(u)
            return Concatenate(axis=-1)([u, skip_input])

        d0 = Input(shape=(None, None, channels))

        d1 = conv2d(d0, self.gf, bn=False)
        d2 = conv2d(d1, self.gf * 2)
        d3 = conv2d(d2, self.gf * 4)
        d4 = conv2d(d3, self.gf * 8)
        d5 = conv2d(d4, self.gf * 8)

        u1 = deconv2d(d5, d4, self.gf * 8)
        u2 = deconv2d(u1, d3, self.gf * 4)
        u3 = deconv2d(u2, d2, self.gf * 2)
        u4 = deconv2d(u3, d1, self.gf)

        u5 = UpSampling2D(size=2)(u4)
        output_img = Conv2D(1, kernel_size=4, strides=1, padding='same', activation='sigmoid')(u5)
        return Model(d0, output_img)

    def build_discriminator(self, channels):
        def d_layer(layer_input, filters, f_size=4, bn=True):
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        img_A = Input(shape=(None, None, 1))
        img_B = Input(shape=(None, None, channels))
        combined_imgs = Concatenate(axis=-1)([img_A, img_B])

        d1 = d_layer(combined_imgs, self.df, bn=False)
        d2 = d_layer(d1, self.df * 2)
        d3 = d_layer(d2, self.df * 4)
        d4 = d_layer(d3, self.df * 8)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)
        return Model(inputs=[img_A, img_B], outputs=validity)

    def compile(self, optimizer=None, loss=None, metrics=None, **kwargs):
        self.metrics = list(metrics or [])

    def _train_step(self, x, y):
        mask = tf.cast(tf.not_equal(y, 2.0), tf.float32)
        y_clean = tf.where(tf.equal(y, 2.0), 0.0, y)
        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
            fake = self.generator(x, training=True)
            d_real = self.discriminator([y_clean, x], training=True)
            d_fake = self.discriminator([fake, x], training=True)
            d_loss = 0.5 * (tf.reduce_mean(tf.square(d_real - 1.0)) + tf.reduce_mean(tf.square(d_fake)))
            l1 = tf.reduce_sum(tf.abs(y_clean - fake) * mask) / (tf.reduce_sum(mask) + 1e-6)
            g_loss = tf.reduce_mean(tf.square(d_fake - 1.0)) + self.LAMBDA * l1
        self.optimizer.apply_gradients(zip(g_tape.gradient(g_loss, self.generator.trainable_variables), self.generator.trainable_variables))
        self.d_optimizer.apply_gradients(zip(d_tape.gradient(d_loss, self.discriminator.trainable_variables), self.discriminator.trainable_variables))
        return {'loss': masked_bce(y, fake), 'g_loss': g_loss, 'd_loss': d_loss}

    def _test_step(self, x, y):
        fake = self.generator(x, training=False)
        for m in self.metrics:
            m.update_state(y, fake)
        return masked_bce(y, fake)

    def fit(self, x, steps_per_epoch=None, validation_data=None, validation_steps=None, epochs=1, validation_freq=1, callbacks=None, **kwargs):
        self.d_optimizer.learning_rate.assign(self.optimizer.learning_rate)
        train_step = tf.function(self._train_step)
        test_step = tf.function(self._test_step)
        self.stop_training = False
        cbs = CallbackList(callbacks, model=self, epochs=epochs, steps=steps_per_epoch, verbose=0)
        train_it = iter(x)
        logs = {}
        cbs.on_train_begin()
        for epoch in range(epochs):
            cbs.on_epoch_begin(epoch)
            for step in range(steps_per_epoch):
                cbs.on_train_batch_begin(step)
                logs = {k: float(v) for k, v in train_step(*next(train_it)).items()}
                cbs.on_train_batch_end(step, logs)
                if self.stop_training:
                    break
            if validation_data is not None and validation_steps and (epoch + 1) % validation_freq == 0:
                for m in self.metrics:
                    m.reset_state()
                val_it = iter(validation_data)
                logs['val_loss'] = sum(float(test_step(*next(val_it))) for _ in range(validation_steps)) / validation_steps
                for m in self.metrics:
                    logs['val_' + m.name] = float(m.result())
            print(f"Epoch {epoch + 1}/{epochs} - " + " - ".join(f"{k}: {v:.4f}" for k, v in logs.items()))
            cbs.on_epoch_end(epoch, logs)
            if self.stop_training:
                break
        cbs.on_train_end(logs)

    def call(self, x, training=False):
        return self.generator(x, training=training)

    def predict(self, images, verbose=0, **kwargs):
        return self.generator.predict(images, verbose=verbose)

    @property
    def input_shape(self):
        return self.generator.input_shape

    @property
    def output_shape(self):
        return self.generator.output_shape

    def count_params(self):
        return self.generator.count_params()

    def get_weights(self):
        return self.generator.get_weights()

    def set_weights(self, weights):
        self.generator.set_weights(weights)

    def save(self, model_path):
        self.generator.save(model_path)

    def load(self, model_path):
        self.generator = tf.keras.models.load_model(model_path, compile=False)
