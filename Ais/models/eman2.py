from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.image import resize
from .losses import masked_bce

title = "Eman2"
include = False

def create(input_shape):
    inputs = Input(input_shape)

    # encoding
    conv1 = Conv2D(40, (15, 15), activation = 'relu', padding='same')(inputs)
    max1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(40, (15, 15), activation = 'relu', padding='same')(max1)
    conv3 = Conv2D(1, (15, 15), activation = 'sigmoid', padding='same')(conv2)
    up4 = UpSampling2D(size=(2, 2))(conv3)

    model = Model(inputs=[inputs], outputs=[up4])
    model.compile(optimizer=Adam(), loss=masked_bce)

    return model


