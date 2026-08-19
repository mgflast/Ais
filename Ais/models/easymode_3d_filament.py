# ezm-3d-filament - a filament-specific 3D slab model with a fixed tube-render head.
#
# A filament is a 1-D centreline plus a KNOWN radius; the tube is a consequence of that radius, not
# something to learn. So the U-Net trunk predicts only where the centreline is, and a FIXED, isotropic
# ball dilation of radius D/2 renders the D-thick tube: T = tanh(conv3d(C, ball)).
#
# The subtle part is Z. The render only ADDS thickness (dilation can't cap it), so if the centreline C
# smears across Z following the missing wedge, the tube comes out elongated - a blob, not a cylinder.
# But we must NOT over-correct to "one Z per column": actin comes in bundles and crossing/junction
# sites, where two filaments genuinely share an XY column at different Z. The right statement is that a
# centreline is a Z-RIDGE - a local maximum of the filament affinity along Z. So the head detects
# Z-local-maxima: presence (sigmoid of the affinity) AND a ridge test - affinity higher than the slice
# above AND below (the minimum of the forward and backward Z-differences, >0 only at a true peak, so the
# rising edges of a smear are not flagged). Their product keeps thin ridges, suppresses the smear on
# either side, and ALLOWS MULTIPLE ridges per column (a single softmax-over-Z could not). Elongation of
# any one filament is gone, bundles/crossings survive, and the render stays a true ball.
#
# Supervision is the ordinary slab pipeline (masked_bce on the annotated slice, Z-jitter). Train on
# binary labels (the render supplies the thickness) or with `ais train --filament D`, which also sets
# the render diameter to D; it otherwise defaults to DEFAULT_DIAMETER_PX. The 'centerline' layer is the
# thin traced ridge C - straight into core/filaments.py picking.

import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv3D, MaxPooling3D, Conv3DTranspose, BatchNormalization,
                                     concatenate, Dropout, Activation, Multiply, Minimum)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import Constant
from .losses import masked_bce


title = "ezm-3d-filament"
include = True
dimensionality = 3

DEFAULT_DIAMETER_PX = 8.0    # filament diameter in px at the training pixel size; overridden by --filament D
RENDER_AMP = 1.5             # ball amplitude: a centreline voxel -> tanh(AMP)~0.9 (crisp tube, gradients alive)
PRESENCE_BIAS = -8.0         # initial bias on the affinity logits. Starts presence near 0 so training begins
                             # near-empty (the isotropic ball sums many voxels, so C must start very small).
RENDER_LAYER = "tube_render"

_ACTIVE_DIAMETER_PX = None


def ball_kernel_weights(diameter_px, amp=RENDER_AMP):
    """Fixed render kernel: a solid ISOTROPIC ball of radius D/2 (value=amp inside, 0 outside), shape
    (kY, kX, kZ, 1, 1). Dilating the thin centreline C by it and squashing with tanh renders a real
    D-thick cylinder - thickness set by the ball, Z-thinness of C guaranteed by the ridge head."""
    R = max(1, int(round(float(diameter_px) / 2.0)))
    n = 2 * R + 1
    ax = np.arange(n) - R
    yy, xx, zz = np.meshgrid(ax, ax, ax, indexing='ij')
    ball = (yy ** 2 + xx ** 2 + zz ** 2 <= R ** 2).astype(np.float32) * float(amp)
    return ball.reshape(n, n, n, 1, 1)


def create(input_shape, output_dimensionality=1):
    input_shape = (*input_shape, 1)
    drop_rate_bottleneck = 0.25
    inputs = Input(input_shape)

    # --- 3D U-Net trunk (identical to ezm-3d: Z pooled at levels 1-2, XY at all levels) ---
    conv1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(conv1)

    conv2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(conv2)

    conv3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 1))(conv3)

    conv4 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    pool4 = MaxPooling3D(pool_size=(2, 2, 1))(conv4)

    conv5 = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)
    drop5 = Dropout(drop_rate_bottleneck)(conv5)

    up6 = Conv3DTranspose(256, (2, 2, 1), strides=(2, 2, 1), padding='same')(drop5)
    merge6 = concatenate([up6, conv4], axis=-1)
    conv6 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(merge6)
    conv6 = BatchNormalization()(conv6)

    up7 = Conv3DTranspose(128, (2, 2, 1), strides=(2, 2, 1), padding='same')(conv6)
    merge7 = concatenate([up7, conv3], axis=-1)
    conv7 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(merge7)
    conv7 = BatchNormalization()(conv7)

    up8 = Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv7)
    merge8 = concatenate([up8, conv2], axis=-1)
    conv8 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(merge8)
    conv8 = BatchNormalization()(conv8)

    up9 = Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv8)
    merge9 = concatenate([up9, conv1], axis=-1)
    conv9 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(merge9)
    conv9 = BatchNormalization()(conv9)

    # --- centreline head: a centreline is a Z-RIDGE - a local maximum of the filament affinity along Z.
    # Detect Z-local-maxima: presence (sigmoid A) AND a ridge test - affinity higher than the slice above
    # AND below. Two fixed first-difference stencils give A(z)-A(z+1) and A(z)-A(z-1); their MINIMUM is >0
    # only at a strict peak (both neighbours lower), so rising edges of a smear are not flagged. Their
    # product keeps thin ridges, suppresses the smear on either side, and allows MULTIPLE ridges per XY
    # column (bundles, crossings). PRESENCE_BIAS starts the affinity low so training begins near-empty.
    logits = Conv3D(output_dimensionality, (1, 1, 1), bias_initializer=Constant(PRESENCE_BIAS),
                    name='cl_logits')(conv9)                          # (Y,X,Z,C) per-Z filament affinity
    d_fwd = Conv3D(output_dimensionality, (1, 1, 3), use_bias=False, padding='same',
                   trainable=False, name='z_diff_fwd')(logits)       # A(z) - A(z+1)
    d_bwd = Conv3D(output_dimensionality, (1, 1, 3), use_bias=False, padding='same',
                   trainable=False, name='z_diff_bwd')(logits)       # A(z) - A(z-1)
    ridge = Minimum()([d_fwd, d_bwd])                                # >0 only at a strict Z-local-max
    centerline = Multiply(name='centerline')([Activation('sigmoid')(logits),
                                              Activation('sigmoid')(ridge)])   # presence AND ridge

    # --- FIXED tube-render: dilate the thin centreline by an ISOTROPIC ball of radius D/2 -> a real
    # D-cylinder. Diameter from --filament D (via _ACTIVE_DIAMETER_PX, set by se_model.compile) or default.
    global _ACTIVE_DIAMETER_PX
    diameter = float(_ACTIVE_DIAMETER_PX) if _ACTIVE_DIAMETER_PX else DEFAULT_DIAMETER_PX
    _default = _ACTIVE_DIAMETER_PX is None
    _ACTIVE_DIAMETER_PX = None                                        # consume: never leak into another build
    R = max(1, int(round(diameter / 2.0)))
    print(f"{title}: isotropic tube render, diameter {diameter:g} px (ball radius {R})"
          + (" - DEFAULT, no --filament given" if _default else ""))
    render = Conv3D(output_dimensionality, (2 * R + 1, 2 * R + 1, 2 * R + 1),
                    use_bias=False, padding='same', trainable=False, name=RENDER_LAYER)(centerline)
    output = Activation('tanh', name='tube')(render)                 # (Y, X, Z, C) tube slab

    model = Model(inputs=[inputs], outputs=[output])
    model.get_layer(RENDER_LAYER).set_weights([ball_kernel_weights(diameter)])
    # fixed first-difference stencils: A(z)-A(z+1) and A(z)-A(z-1); min>0 <=> strict Z-local-max
    model.get_layer('z_diff_fwd').set_weights([np.array([0.0, 1.0, -1.0], np.float32).reshape(1, 1, 3, 1, 1)])
    model.get_layer('z_diff_bwd').set_weights([np.array([-1.0, 1.0, 0.0], np.float32).reshape(1, 1, 3, 1, 1)])

    model.compile(optimizer=Adam(learning_rate=5e-5), loss=masked_bce)
    return model
