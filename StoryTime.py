import numpy as np

from keras.layers import Input
from keras.models import Model
from keras.preprocessing import sequence

from config import CONFIG
import components

class StorytimeArchitecture:

    def __init__(self, seq_length):
        self.main_input(seq_length)
        self.path = components.encoder_embedding(self.main_input)
        self.path = components.encoder_prenet(self.path)
        self.path = components.encoder_cbhg(self.path, self.path)
        print("encoder_output: %s" % self.path.get_shape())
        self.main_output()

    # called first in __init__()
    def main_input(self, seq_length):
        self.main_input = Input(shape=(seq_length, ), dtype='int32')

    # called last in __init__()
    def main_output(self):
        self.main_output = self.path

if __name__ == "__main__":
    # Generate dummy data
    dummy_inputs = []
    dummy_batch_size = 32
    dummy_vocab_size = 256
    dummy_max_length = 50
    for i in range(dummy_batch_size):
        timesteps = np.random.randint(dummy_max_length, size=1)[0]
        batch = np.random.randint(dummy_vocab_size, size=timesteps)
        dummy_inputs.append(batch.tolist())
    inputs = dummy_inputs

    # Pad inputs
    padded_inputs = sequence.pad_sequences(inputs, padding='post', value=0)

    # Build graph
    architecture = StorytimeArchitecture(padded_inputs.shape[-1])
    model = Model(inputs=architecture.main_input, outputs=architecture.main_output)

    # model.compile()

    # # Train the model, iterating on the data in batches of 32 samples
    # model.fit(data, labels, epochs=10, batch_size=32)