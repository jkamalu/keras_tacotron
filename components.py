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

def encoder_prenet(inputs):
    prenet_output = Dense(256, activation='relu')(inputs)
    prenet_output = Dropout(0.5)(prenet_output)
    prenet_output = Dense(128, activation='relu')(prenet_output)
    print("prenet_output: %s" % prenet_output.get_shape())
    return Dropout(0.5)(prenet_output)

def convolutional_bank(inputs):
    convolutions = Conv1D(CONFIG.embed_size // 2, 1)(inputs)
    print("convolutional_bank: %s" % convolutions.get_shape())
    for i in range(2, CONFIG.num_conv_regions + 1):
        conv = Conv1D(CONFIG.embed_size // 2, i)(inputs)
        print("conv: %s" % conv.get_shape())
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
    max_pooling = MaxPooling1D(pool_size=2, strides=1)(convolutions)
    # convolutional projections
    projection = Conv1D(CONFIG.embed_size // 2, 3, activation='relu')(max_pooling)
    projection = Conv1D(CONFIG.embed_size // 2, 3, activation='linear')(projection)
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






