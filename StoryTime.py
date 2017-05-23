from keras import backend as K
from keras.layers import Input
from keras.models import Model

import tensorflow as tf
import numpy as np

import config
import components

class StorytimeArchitecture:

    def __init__(self):
        self.create_placeholders()
        self.main_input()
        self.path = components.encoder_prenet(self.main_input)
        self.path = components.encoder_cbhg(self.path, self.main_input)

    # TODO: missing placeholder from assign 2, feed dict 
    def create_placeholders(self):
        self.seqlen_placeholder = K.placeholder(shape=(None,), dtype='int32')
        self.inputs_placeholder = K.placeholder(shape=(None, None, 256), dtype='int32')

    # TODO: input tensor shape, placeholders
    def main_input(self):
        self.main_input = Input(shape=(None, 256), dtype='float32', name='main_input')

if __name__ == "__main__":
    architecture = StorytimeArchitecture()