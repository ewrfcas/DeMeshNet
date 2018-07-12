# used for DeMesh
from keras.models import *
from keras.layers import *
from keras import layers

def Residual(x, filters):
    # Skip layer
    shortcut = Conv2D(filters, (1, 1), padding='same')(x)

    # Residual block
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(int(filters / 2), (1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(int(filters / 2), (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, (1, 1), padding='same', use_bias=False)(x)
    x = layers.add([x, shortcut])

    return x

def SegNet(image_size=(224, 176, 3)):
    inputs = Input(image_size)
    mask = Input((224, 176, 1))
    x = inputs

    # encoder
    x = Residual(x, 64)
    x = MaxPooling2D()(x)

    x = Residual(x, 128)
    x = MaxPooling2D()(x)

    x = Residual(x, 256)
    x = MaxPooling2D()(x)

    x = Residual(x, 512)
    x = MaxPooling2D()(x)

    # decoder
    x = UpSampling2D()(x)
    x = Residual(x, 512)
    x = BatchNormalization()(x)

    x = UpSampling2D()(x)
    x = Residual(x, 256)
    x = BatchNormalization()(x)

    x = UpSampling2D()(x)
    x = Residual(x, 128)
    x = BatchNormalization()(x)

    x = UpSampling2D()(x)
    x = Residual(x, 64)
    x = BatchNormalization()(x)

    x1 = Conv2D(3, 1, activation='tanh', name='pixel')(x)
    x2 = Multiply(name='mask')([mask, x1])

    return Model(inputs=[inputs, mask], outputs=[x1, x2])

model=SegNet()
model.summary()