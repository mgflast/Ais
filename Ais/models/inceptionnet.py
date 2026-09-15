from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Activation
from tensorflow.keras.optimizers import Adam
from .losses import masked_bce

title = "InceptionNet"
include = False

def inception_module(inputs, filters=64):
    t1 = Conv2D(filters, (1, 1), padding='same', activation='relu')(inputs)

    t2 = Conv2D(filters, (1, 1), padding='same', activation='relu')(inputs)
    t2 = Conv2D(filters, (3, 3), padding='same', activation='relu')(t2)

    t3 = Conv2D(filters, (1, 1), padding='same', activation='relu')(inputs)
    t3 = Conv2D(filters, (5, 5), padding='same', activation='relu')(t3)

    t4 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(inputs)
    t4 = Conv2D(filters, (1, 1), padding='same', activation='relu')(t4)

    return Concatenate()([t1, t2, t3, t4])


def create(input_shape):
    inputs = Input(input_shape)

    # Add multiple inception modules
    inception1 = inception_module(inputs)
    inception2 = inception_module(inception1)
    inception3 = inception_module(inception2)

    # Apply a final convolution
    output = Conv2D(1, (1, 1), padding='same', activation="sigmoid")(inception3)

    # create the model
    model = Model(inputs=[inputs], outputs=[output])
    model.compile(optimizer=Adam(), loss=masked_bce)

    return model