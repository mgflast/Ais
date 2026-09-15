from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Activation, Add
from tensorflow.keras.optimizers import Adam
from .losses import masked_bce

title = "ResNet"
include = False

def create(input_shape):
    inputs = Input(input_shape)

    # Residual block 1
    conv1 = Conv2D(32, (3, 3), padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), padding='same')(conv1)
    shortcut = Conv2D(32, (1, 1), padding='same')(inputs)
    add1 = Add()([shortcut, conv1])
    act1 = Activation('relu')(add1)

    # Residual block 2I
    conv2 = Conv2D(64, (3, 3), padding='same')(act1)
    conv2 = Conv2D(64, (3, 3), padding='same')(conv2)
    shortcut2 = Conv2D(64, (1, 1), padding='same')(act1)
    add2 = Add()([shortcut2, conv2])
    act2 = Activation('relu')(add2)

    # Residual block 3
    conv3 = Conv2D(128, (3, 3), padding='same')(act2)
    conv3 = Conv2D(128, (3, 3), padding='same')(conv3)
    shortcut3 = Conv2D(128, (1, 1), padding='same')(act2)
    add3 = Add()([shortcut3, conv3])
    act3 = Activation('relu')(add3)

    # Residual block 4
    conv4 = Conv2D(256, (3, 3), padding='same')(act3)
    conv4 = Conv2D(256, (3, 3), padding='same')(conv4)
    shortcut4 = Conv2D(256, (1, 1), padding='same')(act3)
    add4 = Add()([shortcut4, conv4])
    act4 = Activation('relu')(add4)

    output = Conv2D(1, (1, 1), activation='sigmoid')(act4)

    # create the model
    model = Model(inputs=[inputs], outputs=[output])
    model.compile(optimizer=Adam(), loss=masked_bce)

    return model
