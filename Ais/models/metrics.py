import tensorflow as tf

include = False


def _center_slice_if_slab(y_true, y_pred):
    # a 3D slab prediction is scored on its central Z slice against the 2D label. y_pred is either
    # (B, Y, X, Z, C) or, if Keras squeezed the trailing size-1 channel, (B, Y, X, Z).
    pr, tr = y_pred.shape.rank, y_true.shape.rank
    if pr is None or tr is None:
        return y_pred
    if pr == tr + 1:                                     # (B, Y, X, Z, 1) vs (B, Y, X, 1)
        zc = tf.shape(y_pred)[-2] // 2
        return y_pred[..., zc, :]
    if pr == tr and y_true.shape[-1] == 1 and y_pred.shape[-1] not in (None, 1):  # channel squeezed
        zc = tf.shape(y_pred)[-1] // 2
        return y_pred[..., zc:zc + 1]
    return y_pred


class MaskedPrecision(tf.keras.metrics.Metric):
    def __init__(self, threshold=0.5, name='masked_precision', **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.tp = self.add_weight(name='tp', initializer='zeros')
        self.fp = self.add_weight(name='fp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = _center_slice_if_slab(y_true, y_pred)
        valid = tf.cast(tf.not_equal(y_true, 2.0), tf.float32)
        y_true_pos = tf.cast(tf.equal(y_true, 1.0), tf.float32) * valid
        y_pred_pos = tf.cast(y_pred > self.threshold, tf.float32) * valid
        self.tp.assign_add(tf.reduce_sum(y_pred_pos * y_true_pos))
        self.fp.assign_add(tf.reduce_sum(y_pred_pos * (1.0 - y_true_pos)))

    def result(self):
        return tf.math.divide_no_nan(self.tp, self.tp + self.fp)

    def reset_state(self):
        self.tp.assign(0.0)
        self.fp.assign(0.0)

    def get_config(self):
        return {**super().get_config(), 'threshold': self.threshold}


class MaskedRecall(tf.keras.metrics.Metric):
    def __init__(self, threshold=0.5, name='masked_recall', **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.tp = self.add_weight(name='tp', initializer='zeros')
        self.fn = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = _center_slice_if_slab(y_true, y_pred)
        valid = tf.cast(tf.not_equal(y_true, 2.0), tf.float32)
        y_true_pos = tf.cast(tf.equal(y_true, 1.0), tf.float32) * valid
        y_pred_pos = tf.cast(y_pred > self.threshold, tf.float32) * valid
        self.tp.assign_add(tf.reduce_sum(y_pred_pos * y_true_pos))
        self.fn.assign_add(tf.reduce_sum((1.0 - y_pred_pos) * y_true_pos))

    def result(self):
        return tf.math.divide_no_nan(self.tp, self.tp + self.fn)

    def reset_state(self):
        self.tp.assign(0.0)
        self.fn.assign(0.0)

    def get_config(self):
        return {**super().get_config(), 'threshold': self.threshold}
