import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, Conv3DTranspose, BatchNormalization, concatenate, Dropout
from tensorflow.keras.optimizers import Adam
from .losses import masked_bce_dice


title = "ezm-3d-L"
include = True
dimensionality = 3


def factored_conv(x, filters):
    # (3,3,1) + (1,1,3): same receptive field as (3,3,3) at ~1/3 the weights.
    x = Conv3D(filters, (3, 3, 1), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv3D(filters, (1, 1, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    return x


def conv_xy(x, filters):
    x = Conv3D(filters, (3, 3, 1), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    return x


def create(input_shape, output_dimensionality=1):
    input_shape = (*input_shape, 1)
    drop_rate_bottleneck = 0.25
    inputs = Input(input_shape)

    # ezm-3d's slab layout (Z pooled at levels 1-2, XY at all levels) at ezm-2d-dice's
    # in-plane layout: 3 convs per encoder block, 2 per decoder level. Full 3x3x3 kernels
    # only at levels 1-2 where the Z grid is still deep; 512 filter cap at levels 3-5. The
    # Z receptive field saturates by level 2 for typical slabs, so deep levels get one
    # (1,1,3) Z-mixing conv each (keeps the slab output Z-coupled), the rest are (3,3,1).
    conv1 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(conv1)

    conv2 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(conv2)

    conv3 = factored_conv(pool2, 256)
    conv3 = conv_xy(conv3, 256)
    conv3 = conv_xy(conv3, 256)
    pool3 = MaxPooling3D(pool_size=(2, 2, 1))(conv3)

    conv4 = factored_conv(pool3, 512)
    conv4 = conv_xy(conv4, 512)
    conv4 = conv_xy(conv4, 512)
    pool4 = MaxPooling3D(pool_size=(2, 2, 1))(conv4)

    conv5 = factored_conv(pool4, 512)
    conv5 = conv_xy(conv5, 512)
    conv5 = conv_xy(conv5, 512)
    drop5 = Dropout(drop_rate_bottleneck)(conv5)

    up6 = Conv3DTranspose(512, (2, 2, 1), strides=(2, 2, 1), padding='same')(drop5)
    merge6 = concatenate([up6, conv4], axis=-1)
    conv6 = factored_conv(merge6, 512)
    conv6 = conv_xy(conv6, 512)

    up7 = Conv3DTranspose(256, (2, 2, 1), strides=(2, 2, 1), padding='same')(conv6)
    merge7 = concatenate([up7, conv3], axis=-1)
    conv7 = factored_conv(merge7, 256)
    conv7 = conv_xy(conv7, 256)

    up8 = Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv7)
    merge8 = concatenate([up8, conv2], axis=-1)
    conv8 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(merge8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv8)
    conv8 = BatchNormalization()(conv8)

    up9 = Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv8)
    merge9 = concatenate([up9, conv1], axis=-1)
    conv9 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(merge9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv9)
    conv9 = BatchNormalization()(conv9)

    output = Conv3D(output_dimensionality, (1, 1, 1), activation='sigmoid')(conv9)   # (Y, X, Z, C) slab

    model = Model(inputs=[inputs], outputs=[output])
    model.compile(optimizer=Adam(learning_rate=5e-5),
                  loss=masked_bce_dice(bce_weight=1.0, dice_weight=1.0))
    return model
