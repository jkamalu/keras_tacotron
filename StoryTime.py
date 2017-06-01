from keras.layers import Input
from keras.models import Model

import config
import components

class StorytimeArchitecture:

    def __init__(self):
        self.main_input()
        self.path = components.encoder_prenet(self.main_input)
        self.path = components.encoder_cbhg(self.path, self.path)
        print("encoder_output: %s" % self.path.get_shape())
        self.main_output = self.path

    def main_input(self):
        self.main_input = Input(shape=(30, 256), dtype='float32')

if __name__ == "__main__":
    architecture = StorytimeArchitecture()
    model = Model(inputs=architecture.main_input, outputs=architecture.main_output)

    # model.compile()
    # # Generate dummy data
    # import numpy as np
    # data = np.random.random((1000, 100))
    # labels = np.random.randint(2, size=(1000, 1))

    # # Train the model, iterating on the data in batches of 32 samples
    # model.fit(data, labels, epochs=10, batch_size=32)