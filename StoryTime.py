from keras import backend as K

from keras.layers import Input
from keras.layers.core import Dense, Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling1D

from keras.models import Model
from keras.engine.topology import Layer

import tensorflow as tf
import numpy as np

# TODO: Investigate feed dict
# TODO: Define input life cycle
# TODO: Define input/output formatting

class CONFIG:

    batch_size = 32
    num_conv_regions = 12
    num_conv_filters = 128

class StorytimeArchitecture:

    def __init__(self):
        self.create_placeholders()
        self.main_input()
        self.encoder_prenet()
        self.encoder_cbhg()

    def create_placeholders(self):
        self.seqlen_placeholder = K.placeholder(shape=(None,), dtype='int32')
        self.inputs_placeholder = K.placeholder(shape=(None, None, 256), dtype='int32')

    def main_input(self):
        self.main_input = Input(shape=(None, 256), dtype='float32', name='main_input')

    def encoder_prenet(self):
        prenet_output = Dense(256, activation='relu')(self.main_input)
        prenet_output = Dropout(0.5)(prenet_output)
        prenet_output = Dense(128, activation='relu')(prenet_output)
        self.prenet_output = Dropout(0.5)(prenet_output)

    def encoder_cbhg(self):
        convolutions = Conv1D(CONFIG.num_conv_filters, 1)(self.prenet_output)
        for i in range(2, CONFIG.num_conv_regions + 1):
            conv = Conv1D(CONFIG.num_conv_filters, i)(self.prenet_output)
            norm = BatchNormalization()(conv)
            tf.concat((convolutions, conv), -1)
        max_pooling = MaxPooling1D(pool_size=2, strides=1)


if __name__ == "__main__":
    architecture = StorytimeArchitecture()






