from keras.layers import *
from keras.models import *
from keras import layers
from keras.optimizers import *
from skimage import io

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

def Hourglass(x, level, module, filters):
    # up layer
    for i in range(module):
        x = Residual(x, filters)
    up = x

    # low layer
    low = MaxPooling2D()(x)
    for i in range(module):
        low = Residual(low, filters)
    if level>1:
        low = Hourglass(low, level-1, module, filters)
    else:
        for i in range(module):
            low = Residual(low, filters)
    for i in range(module):
        low = Residual(low, filters)
    low = UpSampling2D()(low)
    x = layers.add([up, low])

    return x

def model(input_shape=(224, 176, 3), nstack=2, level=4, module=1, filters=128):
    img_input = Input(shape=input_shape)
    mask = Input((224, 176, 1))
    middle_x = Residual(img_input, filters)
    outputs=[]

    for i in range(nstack):
        x = Hourglass(middle_x, level, module, filters)
        for j in range(module):
            x = Residual(x, filters)
        x = Conv2D(filters, (1, 1), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        temp_output = Conv2D(3, (1, 1), activation='tanh', padding='same', name='nstack_'+str(i+1))(x)
        temp_output_m = Multiply(name='mask_'+str(i+1))([mask, temp_output])
        outputs.append(temp_output)
        outputs.append(temp_output_m)

        if i < nstack-1:
            x = Conv2D(filters, (1, 1), padding='same')(x)
            temp_output = Conv2D(filters, (1, 1), padding='same')(temp_output)
            # temp_output_m = Conv2D(filters, (1, 1), padding='same')(temp_output_m) # 测试的时候随机的mask会破坏下一层的输出
            middle_x = layers.add([middle_x, x, temp_output])# temp_output_m

    # Create model.
    model = Model([img_input, mask], outputs, name='hourglass')

    return model