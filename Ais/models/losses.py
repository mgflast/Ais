# Shared loss functions for the Ais default model library.
#
# Every loss here understands the optional 'ignore' label that the annotation
# exporter can write next to background (0) and foreground (1): pixels equal to
# `ignore_label` (2 by default) are masked out so they contribute neither to the
# loss value nor to its gradient. When no ignore pixels are present each loss
# reduces to its ordinary per-pixel form, so these are safe to use on data
# exported without ignore labels too.
#
# This file is NOT a model. It is still picked up by SEModel.load_models()'s glob
# over models/*.py and imported, but `include = False` keeps it out of the model
# inventory (the same mechanism used by model_template.py and __init__.py).
# Default-library model files pull these in with `from .losses import ...`.

import tensorflow as tf

include = False


def _crop_border(y_true, y_pred, border):
    if border > 0:
        y_true = y_true[:, border:-border, border:-border, ...]
        y_pred = y_pred[:, border:-border, border:-border, ...]
    return y_true, y_pred


def masked_bce(y_true, y_pred, border=0, ignore_label=2.0, epsilon=1e-6):
    """Binary cross-entropy averaged over the non-ignore pixels.

    `border` optionally drops a margin from the loss; it defaults to 0 because the
    annotation exporter already writes ignore labels wherever the edge should be
    excluded. Pass border>0 only if you want to crop on top of that.
    """
    y_true, y_pred = _crop_border(y_true, y_pred, border)
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    mask = tf.cast(tf.not_equal(y_true, ignore_label), tf.float32)
    y_true_clean = tf.where(tf.equal(y_true, ignore_label), 0.0, y_true)

    bce = tf.keras.losses.binary_crossentropy(y_true_clean, y_pred)  # drops the channel axis
    mask_bce = tf.squeeze(mask, axis=-1)   # match bce; works for 2D (B,H,W,1) and slab (B,H,W,D,1)

    return tf.reduce_sum(bce * mask_bce) / (tf.reduce_sum(mask_bce) + epsilon)


def masked_dice(y_true, y_pred, border=0, ignore_label=2.0, epsilon=1e-6):
    """Soft Dice loss computed over the non-ignore pixels (see masked_bce re: border)."""
    y_true, y_pred = _crop_border(y_true, y_pred, border)
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    mask = tf.cast(tf.not_equal(y_true, ignore_label), tf.float32)
    y_true_clean = tf.where(tf.equal(y_true, ignore_label), 0.0, y_true)

    y_true_f = tf.reshape(y_true_clean * mask, [-1])
    y_pred_f = tf.reshape(y_pred * mask, [-1])
    valid = tf.reduce_sum(tf.reshape(mask, [-1]))

    numerator = 2.0 * tf.reduce_sum(y_true_f * y_pred_f)
    denominator = tf.reduce_sum(y_true_f + y_pred_f)

    dice_coeff = tf.where(tf.equal(valid, 0.0), 1.0, (numerator + epsilon) / (denominator + epsilon))
    return 1.0 - dice_coeff


def masked_bce_dice(bce_weight=0.1, dice_weight=1.0, border=0, ignore_label=2.0, epsilon=1e-6):
    """Return a Keras loss = bce_weight * masked_bce + dice_weight * masked_dice.

    The optional border crop (default off) is applied once here, so the wrapped
    losses are called with border=0 to avoid cropping twice.
    """
    def loss(y_true, y_pred):
        yt, yp = _crop_border(y_true, y_pred, border)
        bce = masked_bce(yt, yp, border=0, ignore_label=ignore_label, epsilon=epsilon)
        dice = masked_dice(yt, yp, border=0, ignore_label=ignore_label, epsilon=epsilon)
        return bce_weight * bce + dice_weight * dice

    loss.__name__ = "masked_bce_dice"
    return loss
