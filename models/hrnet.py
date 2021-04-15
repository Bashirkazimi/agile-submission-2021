"""
Tensorflow keras implementation of hrnet
https://github.com/Bashirkazimi/BashirLearning
Author: Bashir Kazimi
Majority of the code borrowed from https://github.com/niecongchong/HRNet-keras-semantic-segmentation
"""

from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Dense
from keras.layers import UpSampling2D, add, concatenate, Add, GlobalAveragePooling2D, MaxPooling2D, Concatenate

import numpy as np
import random 
import tensorflow as tf 

np.random.seed(42)
tf.set_random_seed(42)
random.seed(42)




def conv3x3(x, out_filters, strides=(1, 1)):
    x = Conv2D(out_filters, 3, padding='same', strides=strides, use_bias=False, kernel_initializer='he_normal')(x)
    return x


def basic_Block(input, out_filters, strides=(1, 1), with_conv_shortcut=False):
    x = conv3x3(input, out_filters, strides)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = conv3x3(x, out_filters)
    x = BatchNormalization(axis=3)(x)

    if with_conv_shortcut:
        residual = Conv2D(out_filters, 1, strides=strides, use_bias=False, kernel_initializer='he_normal')(input)
        residual = BatchNormalization(axis=3)(residual)
        x = add([x, residual])
    else:
        x = add([x, input])

    x = Activation('relu')(x)
    return x


def bottleneck_Block(input, out_filters, strides=(1, 1), with_conv_shortcut=False):
    expansion = 4
    de_filters = int(out_filters / expansion)

    x = Conv2D(de_filters, 1, use_bias=False, kernel_initializer='he_normal')(input)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = Conv2D(de_filters, 3, strides=strides, padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = Conv2D(out_filters, 1, use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=3)(x)

    if with_conv_shortcut:
        residual = Conv2D(out_filters, 1, strides=strides, use_bias=False, kernel_initializer='he_normal')(input)
        residual = BatchNormalization(axis=3)(residual)
        x = add([x, residual])
    else:
        x = add([x, input])

    x = Activation('relu')(x)
    return x


def stem_net(input):
    x = Conv2D(64, 3, strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(input)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = bottleneck_Block(x, 256, with_conv_shortcut=True)
    x = bottleneck_Block(x, 256, with_conv_shortcut=False)

    return x

def transition_layer1(x, out_filters_list=[32, 64]):
    x0 = Conv2D(out_filters_list[0], 3, padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x0 = BatchNormalization(axis=3)(x0)
    x0 = Activation('relu')(x0)

    x1 = Conv2D(out_filters_list[1], 3, strides=(2, 2),
                padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x1 = BatchNormalization(axis=3)(x1)
    x1 = Activation('relu')(x1)

    return [x0, x1]


def make_branch1_0(x, out_filters=32):
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    return x


def make_branch1_1(x, out_filters=64):
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    return x


def fuse_layer1(x):
    x0_0 = x[0]
    x0_1 = Conv2D(32, 1, use_bias=False, kernel_initializer='he_normal')(x[1])
    x0_1 = BatchNormalization(axis=3)(x0_1)
    x0_1 = UpSampling2D(size=(2, 2),interpolation='bilinear')(x0_1)
    x0 = add([x0_0, x0_1])

    x1_0 = Conv2D(64, 3, strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(x[0])
    x1_0 = BatchNormalization(axis=3)(x1_0)
    x1_1 = x[1]
    x1 = add([x1_0, x1_1])
    return [x0, x1]


def transition_layer2(x, out_filters_list=[32, 64, 128]):
    x0 = Conv2D(out_filters_list[0], 3, padding='same', use_bias=False, kernel_initializer='he_normal')(x[0])
    x0 = BatchNormalization(axis=3)(x0)
    x0 = Activation('relu')(x0)

    x1 = Conv2D(out_filters_list[1], 3, padding='same', use_bias=False, kernel_initializer='he_normal')(x[1])
    x1 = BatchNormalization(axis=3)(x1)
    x1 = Activation('relu')(x1)

    x2 = Conv2D(out_filters_list[2], 3, strides=(2, 2),
                padding='same', use_bias=False, kernel_initializer='he_normal')(x[1])
    x2 = BatchNormalization(axis=3)(x2)
    x2 = Activation('relu')(x2)

    return [x0, x1, x2]


def make_branch2_0(x, out_filters=32):
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    return x


def make_branch2_1(x, out_filters=64):
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    return x


def make_branch2_2(x, out_filters=128):
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    return x


def fuse_layer2(x):
    x0_0 = x[0]
    x0_1 = Conv2D(32, 1, use_bias=False, kernel_initializer='he_normal')(x[1])
    x0_1 = BatchNormalization(axis=3)(x0_1)
    x0_1 = UpSampling2D(size=(2, 2),interpolation='bilinear')(x0_1)
    x0_2 = Conv2D(32, 1, use_bias=False, kernel_initializer='he_normal')(x[2])
    x0_2 = BatchNormalization(axis=3)(x0_2)
    x0_2 = UpSampling2D(size=(4, 4),interpolation='bilinear')(x0_2)
    x0 = add([x0_0, x0_1, x0_2])

    x1_0 = Conv2D(64, 3, strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(x[0])
    x1_0 = BatchNormalization(axis=3)(x1_0)
    x1_1 = x[1]
    x1_2 = Conv2D(64, 1, use_bias=False, kernel_initializer='he_normal')(x[2])
    x1_2 = BatchNormalization(axis=3)(x1_2)
    x1_2 = UpSampling2D(size=(2, 2),interpolation='bilinear')(x1_2)
    x1 = add([x1_0, x1_1, x1_2])

    x2_0 = Conv2D(32, 3, strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(x[0])
    x2_0 = BatchNormalization(axis=3)(x2_0)
    x2_0 = Activation('relu')(x2_0)
    x2_0 = Conv2D(128, 3, strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(x2_0)
    x2_0 = BatchNormalization(axis=3)(x2_0)
    x2_1 = Conv2D(128, 3, strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(x[1])
    x2_1 = BatchNormalization(axis=3)(x2_1)
    x2_2 = x[2]
    x2 = add([x2_0, x2_1, x2_2])
    return [x0, x1, x2]


def transition_layer3(x, out_filters_list=[32, 64, 128, 256]):
    x0 = Conv2D(out_filters_list[0], 3, padding='same', use_bias=False, kernel_initializer='he_normal')(x[0])
    x0 = BatchNormalization(axis=3)(x0)
    x0 = Activation('relu')(x0)

    x1 = Conv2D(out_filters_list[1], 3, padding='same', use_bias=False, kernel_initializer='he_normal')(x[1])
    x1 = BatchNormalization(axis=3)(x1)
    x1 = Activation('relu')(x1)

    x2 = Conv2D(out_filters_list[2], 3, padding='same', use_bias=False, kernel_initializer='he_normal')(x[2])
    x2 = BatchNormalization(axis=3)(x2)
    x2 = Activation('relu')(x2)

    x3 = Conv2D(out_filters_list[3], 3, strides=(2, 2),
                padding='same', use_bias=False, kernel_initializer='he_normal')(x[2])
    x3 = BatchNormalization(axis=3)(x3)
    x3 = Activation('relu')(x3)

    return [x0, x1, x2, x3]


def make_branch3_0(x, out_filters=32):
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    return x


def make_branch3_1(x, out_filters=64):
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    return x


def make_branch3_2(x, out_filters=128):
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    return x


def make_branch3_3(x, out_filters=256):
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    return x


def fuse_layer3(x):
    x0_0 = x[0]
    x0_1 = Conv2D(32, 1, use_bias=False, kernel_initializer='he_normal')(x[1])
    x0_1 = BatchNormalization(axis=3)(x0_1)
    x0_1 = UpSampling2D(size=(2, 2),interpolation='bilinear')(x0_1)
    x0_2 = Conv2D(32, 1, use_bias=False, kernel_initializer='he_normal')(x[2])
    x0_2 = BatchNormalization(axis=3)(x0_2)
    x0_2 = UpSampling2D(size=(4, 4),interpolation='bilinear')(x0_2)
    x0_3 = Conv2D(32, 1, use_bias=False, kernel_initializer='he_normal')(x[3])
    x0_3 = BatchNormalization(axis=3)(x0_3)
    x0_3 = UpSampling2D(size=(8, 8),interpolation='bilinear')(x0_3)
    x0 = concatenate([x0_0, x0_1, x0_2, x0_3], axis=-1)
    return x0


def feature_extractor1(inputs):

    x = stem_net(inputs)

    x = transition_layer1(x)
    x0 = make_branch1_0(x[0])
    x1 = make_branch1_1(x[1])
    x = fuse_layer1([x0, x1])

    phase1 = Model(inputs=inputs, outputs=x, name='phase1')

    x = transition_layer2(x)
    x0 = make_branch2_0(x[0])
    x1 = make_branch2_1(x[1])
    x2 = make_branch2_2(x[2])
    x = fuse_layer2([x0, x1, x2])

    phase2 = Model(inputs=inputs, outputs=x, name='phase2')

    x = transition_layer3(x)
    x0 = make_branch3_0(x[0])
    x1 = make_branch3_1(x[1])
    x2 = make_branch3_2(x[2])
    x3 = make_branch3_3(x[3])
    x = fuse_layer3([x0, x1, x2, x3])

    phase3 = Model(inputs=inputs, outputs=x, name='phase3')

    return x, phase1, phase2, phase3


def hrnet_classifier(input_shape=(128, 128, 1), classes=5, activation='softmax'):
    inputs = Input(shape=input_shape)
    x, phase1, phase2, phase3 = feature_extractor1(inputs)
    
    x = basic_Block(x, 128, with_conv_shortcut=False)
    x = basic_Block(x, 128, with_conv_shortcut=False)
    x = basic_Block(x, 128, with_conv_shortcut=False)
    x = basic_Block(x, 128, with_conv_shortcut=False)
    
    x = GlobalAveragePooling2D()(x)

    out = Dense(classes, activation=activation)(x)

    model = Model(inputs=inputs, outputs=out)
    model.summary()

    return model, phase1, phase2, phase3


def hrnet_segmenter(input_shape=(224, 224, 3), classes=5, halfed=True, activation='softmax'):
    """
    Returns a keras HRNet model for segmentation based on https://arxiv.org/abs/1908.07919
    Args:
        input_shape: input shape
        classes: number of categories

    Returns: keras Model instance

    """
    inputs = Input(shape=input_shape)
    x, phase1, phase2, phase3 = feature_extractor1(inputs)
    
    x = basic_Block(x, 128, with_conv_shortcut=False)
    x = basic_Block(x, 128, with_conv_shortcut=False)
    x = basic_Block(x, 128, with_conv_shortcut=False)
    x = basic_Block(x, 128, with_conv_shortcut=False)

    if not halfed:
        x = UpSampling2D(size=(2, 2),interpolation='bilinear')(x)
        x = Conv2D(classes, 1, use_bias=False, kernel_initializer='he_normal')(x)
        x = BatchNormalization(axis=3)(x)
    else:
        x = Conv2D(classes, 1, use_bias=False, kernel_initializer='he_normal')(x)
        x = BatchNormalization(axis=3)(x)

    out = Activation(activation, name='out')(x)

    model = Model(inputs=inputs, outputs=out)
    model.summary()

    return model, phase1, phase2, phase3


def feature_extractor2(inputs):

    C1 = stem_net(inputs)
     
    x = transition_layer1(C1)

    C1 = MaxPooling2D((2,2))(C1)

    x0 = make_branch1_0(x[0])
    x1 = make_branch1_1(x[1])
    C2 = x1
    x = fuse_layer1([x0, x1])

    phase1 = Model(inputs=inputs, outputs=x, name='phase1')

    x = transition_layer2(x)

    x0 = make_branch2_0(x[0])
    x1 = make_branch2_1(x[1])
    x2 = make_branch2_2(x[2])
    C3 = x2
    x = fuse_layer2([x0, x1, x2])
    
    phase2 = Model(inputs=inputs, outputs=x, name='phase2')

    x = transition_layer3(x)

    x0 = make_branch3_0(x[0])
    x1 = make_branch3_1(x[1])
    x2 = make_branch3_2(x[2])
    x3 = make_branch3_3(x[3])
    C4 = x3

    x = fuse_layer3([x0, x1, x2, x3])
    
    C5 = MaxPooling2D((16, 16))(x)

    phase3 = Model(inputs=inputs, outputs=x, name='phase3')

    mrcnn_fe = Model(inputs=inputs, outputs=[C2, C3, C4, C5], name='mrcnn_fe')
    return x, phase1, phase2, mrcnn_fe , phase3


def hrnet_encoder_decoder(input_shape=(224, 224, 3), output_channels=1, halfed=True):

    inputs = Input(shape=input_shape)
    x, phase1, phase2, phase_2half, phase3 = feature_extractor2(inputs)
    
    x = basic_Block(x, 128, with_conv_shortcut=False)
    x = basic_Block(x, 128, with_conv_shortcut=False)
    x = basic_Block(x, 128, with_conv_shortcut=False)
    x = basic_Block(x, 128, with_conv_shortcut=False)

    if not halfed:
        x = UpSampling2D(size=(2, 2),interpolation='bilinear')(x)
        x = Conv2D(output_channels, 1, use_bias=False, kernel_initializer='he_normal')(x)
        x = BatchNormalization(axis=3)(x)
    else:
        x = Conv2D(output_channels, 1, use_bias=False, kernel_initializer='he_normal')(x)
        x = BatchNormalization(axis=3)(x)

    out = Activation('sigmoid', name='out')(x)

    model = Model(inputs=inputs, outputs=out)
    model.summary()

    return model, phase1, phase2, phase_2half, phase3






