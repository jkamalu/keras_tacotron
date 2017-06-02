import tensorflow as tf

from keras.layers.core import Dense, Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling1D
from keras.layers.recurrent import GRU
from keras.layers.wrappers import Bidirectional
from keras.layers.merge import Add, Multiply
from keras.initializers import Constant

from config import CONFIG

def attention(inputs, memory, num_units=256):
    # Tensorflow underlying code to support Bahdanau attention
    # Returns a tensorflow
    def attend(inp, mem):
        # The attention component will be in control of attending to the given memory
        attention = tf.contrib.seq2seq.BahdanauAttention(num_units, mem)
        cell = tf.contrib.rnn.GRUCell(num_units)

        cell_with_attention = tf.contrib.seq2seq.DynamicAttentionWrapper(cell, attention, num_units)
        outputs, _ = tf.nn.dynamic_rnn(cell_with_attention, inp, dtype=tf.float32)
        return outputs

    # output should be [batches, timesteps, num_units]
    return Lambda(attend, output_shape=(b,t,num_units))(inputs, memory)


def encoder_prenet(inputs):
    prenet_output = Dense(256, activation='relu')(inputs)
    prenet_output = Dropout(0.5)(prenet_output)
    prenet_output = Dense(128, activation='relu')(prenet_output)
    print("prenet_output: %s" % prenet_output.get_shape())
    return Dropout(0.5)(prenet_output)

def convolutional_bank(inputs):
    convolutions = Conv1D(CONFIG.embed_size // 2, 1, padding='same')(inputs)
    for i in range(2, CONFIG.num_conv_regions + 1):
        conv = Conv1D(CONFIG.embed_size // 2, i, padding='same')(inputs)
        norm = BatchNormalization()(conv)
        convolutions = tf.concat((convolutions, conv), -1)
    print("convolutional_bank: %s" % convolutions.get_shape())
    return convolutions

def highway_network(inputs, num_layers=1):
    # https://arxiv.org/pdf/1505.00387.pdf
    # output = H(input,WH) * T(input,WT) + input * C(x,WC)
    layer_inputs = inputs
    for i in range(num_layers):
        H = Dense(128, activation='relu')(layer_inputs)
        T = Dense(128, activation='sigmoid', use_bias=True, bias_initializer=Constant(-1))(layer_inputs)
        C = 1. - T
        layer_inputs = H * T + inputs * C
    print("highway_network: %s" % layer_inputs.get_shape())
    return layer_inputs

def encoder_cbhg(inputs, residual_input=None):
    # convolutional bank
    convolutions = convolutional_bank(inputs)
    # max pooling
    max_pooling = MaxPooling1D(pool_size=2, strides=1, padding='same')(convolutions)
    print("max_pooling: %s" % max_pooling.get_shape())
    # convolutional projections
    projection = Conv1D(CONFIG.embed_size // 2, 3, padding='same', activation='relu')(max_pooling)
    projection = Conv1D(CONFIG.embed_size // 2, 3, padding='same', activation='linear')(projection)
    # residual connection
    print("projection, residual: %s, %s" % (projection.get_shape(), residual_input.get_shape()))
    if residual_input is not None:
        residual = Add()([projection, residual_input])
    else:
        residual = projection
    # highway network
    highway = highway_network(residual, 4)
    # bidirectional gru
    bidirectional_gru = Bidirectional(GRU(128))(highway)
    print("encoder_cbhg: %s" % bidirectional_gru.get_shape())
    return bidirectional_gru
