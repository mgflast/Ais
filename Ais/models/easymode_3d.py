import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, Conv3DTranspose, BatchNormalization, concatenate, Dropout
from tensorflow.keras.optimizers import Adam
from .losses import masked_bce_dice


title = "ezm-3d"
include = True
dimensionality = 3


def create(input_shape, output_dimensionality=1):
    input_shape = (*input_shape, 1)
    drop_rate_bottleneck = 0.25
    inputs = Input(input_shape)

    # 3D encoder (unchanged: Z pooled at levels 1-2, XY at all levels)
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

    # 3D decoder: keep Z, mirror the encoder's pooling, full 3D skips (no center-slice collapse).
    # Adjacent output slices pass through the same Z-mixing convs -> Z-coupled slab output.
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

    output = Conv3D(output_dimensionality, (1, 1, 1), activation='sigmoid')(conv9)   # (Y, X, Z, C) slab

    model = Model(inputs=[inputs], outputs=[output])
    # slab output + slab label (annotated slice at its jittered Z-position, rest = ignore): the
    # masked loss supervises only that slice, and Z-jitter spreads it across all output positions.
    model.compile(optimizer=Adam(learning_rate=5e-5),
                  loss=masked_bce_dice(bce_weight=0.3, dice_weight=0.7))
    return model
