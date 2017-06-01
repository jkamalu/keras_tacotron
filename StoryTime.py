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
        self.path = components.encoder_cbhg(self.path, self.path)
        self.main_output = self.path

    # TODO: missing placeholder from assign 2, feed dict 
    def create_placeholders(self):
        self.seqlen_placeholder = K.placeholder(shape=(None,), dtype='int32')
        self.inputs_placeholder = K.placeholder(shape=(None, None, 256), dtype='int32')

    # TODO: input tensor shape, placeholders
    def main_input(self):
        self.main_input = Input(shape=(30, 256), dtype='float32')
        print("main_input: %s" % self.main_input.get_shape())

if __name__ == "__main__":
    architecture = StorytimeArchitecture()

    model = Model(inputs=architecture.main_input, outputs=architecture.main_output)
    #model.compile()
    # # Generate dummy data
    # import numpy as np
    # data = np.random.random((1000, 100))
    # labels = np.random.randint(2, size=(1000, 1))

    # # Train the model, iterating on the data in batches of 32 samples
    # model.fit(data, labels, epochs=10, batch_size=32)