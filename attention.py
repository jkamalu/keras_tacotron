from keras import backend as K
from keras.engine.topology import Layer
from keras.models import Model
import numpy as np
import tensorflow as tf

class BahdanauAttention(Layer):

    # The Bahdanau attention layer has to attend to a particular set of memory states
    # These are usually the output of some encoder process, where we take the output of
    # GRU states
    def __init__(self, memory, num_units, **kwargs):
        self.memory = memory
        self.num_units = num_units
        super(BahdanauAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        # The attention component will be in control of attending to the given memory
        attention = tf.contrib.seq2seq.BahdanauAttention(self.num_units, self.memory)
        cell = tf.contrib.rnn.GRUCell(num_units)

        cell_with_attention = tf.contrib.seq2seq.DynamicAttentionWrapper(cell, attention, num_units)
        self.outputs, _ = tf.nn.dynamic_rnn(cell_with_attention, inputs, dtype=tf.float32)

        super(MyLayer, self).build(input_shape)

    def call(self, x):
        return

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.memory[1], self.num_units)

inp = np.arange(3*4*8).reshape(3,4,8)
memory = np.arange(3*4*5).reshape(3,4,5)
num_units = 10

attention = BahdanauAttention(memory, num_units)
model = Model(inputs=inp, outputs=attention.outputs)
