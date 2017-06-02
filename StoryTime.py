from keras.layers import Input
from keras.models import Model

import config
import components

class StorytimeArchitecture:

    def __init__(self):
        self.main_input()
        self.path = components.encoder_prenet(self.main_input)
        self.path = components.encoder_cbhg(self.path, self.path)
        self.main_output = self.path

    def main_input(self):
        self.main_input = Input(shape=(30, 256), dtype='float32')

if __name__ == "__main__":
    architecture = StorytimeArchitecture()
    model = Model(inputs=architecture.main_input, outputs=architecture.main_output)


    #Running the model
    #Paper uses the adam optimizer and l1 loss which is based on absolute error
    #we are tracking accuracy metrics, this does not influence the model's training though
    #loss_weights is a non-specified quantity in the paper but it must be the same whenever it is used
        #thus the reason for the use of the config variable, should be in list format
    model.compile(optimizer='adam', loss = 'mean_absolute_error', metrics=['accuracy'], loss_weights=config.loss_weights)

    # model.compile()
    # # Generate dummy data
    # import numpy as np
    # data = np.random.random((1000, 100))
    # labels = np.random.randint(2, size=(1000, 1))

    # # Train the model, iterating on the data in batches of 32 samples
    # model.fit(data, labels, epochs=10, batch_size=32)
