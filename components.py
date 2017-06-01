import tensorflow as tf

from keras.layers import Dense, Dropout, Conv1D, BatchNormalization, MaxPooling1D, GRU, Bidirectional, Lambda
from keras.layers.merge import Add, Multiply, Concatenate
from keras.initializers import Constant

from config import CONFIG

def encoder_prenet(inputs):
    prenet_output = Dense(CONFIG.embed_size, activation='relu')(inputs)
    prenet_output = Dropout(0.5)(prenet_output)
    prenet_output = Dense(CONFIG.embed_size // 2, activation='relu')(prenet_output)
    prenet_output = Dropout(0.5)(prenet_output)
    return prenet_output

def convolutional_bank(inputs):
    convolutions = Conv1D(CONFIG.embed_size // 2, 1, padding='same')(inputs)
    for i in range(2, CONFIG.num_conv_regions + 1):
        conv = Conv1D(CONFIG.embed_size // 2, i, padding='same')(inputs)
        norm = BatchNormalization()(conv)
        convolutions = Concatenate()([convolutions, conv])
    return convolutions

def highway_network(inputs, num_layers=1):
    # https://arxiv.org/pdf/1505.00387.pdf
    # output = H(input,WH) * T(input,WT) + input * C(x,WC)
    layer_inputs = inputs
    for i in range(num_layers):
        H = Dense(CONFIG.embed_size // 2, activation='relu')(layer_inputs)
        T = Dense(CONFIG.embed_size // 2, activation='sigmoid', use_bias=True, bias_initializer=Constant(-1))(layer_inputs)
        C = Lambda(lambda x: 1. - x)(T)
        layer_inputs = Add()([Multiply()([H, T]), Multiply()([inputs, C])])
    return layer_inputs

def encoder_cbhg(inputs, residual_input=None):
    # convolutional bank
    convolutions = convolutional_bank(inputs)
    # max pooling
    max_pooling = MaxPooling1D(pool_size=2, strides=1, padding='same')(convolutions)
    # convolutional projections
    projection = Conv1D(CONFIG.embed_size // 2, 3, padding='same', activation='relu')(max_pooling)
    projection = Conv1D(CONFIG.embed_size // 2, 3, padding='same', activation='linear')(projection)
    # residual connection
    if residual_input is not None: 
        residual = Add()([projection, residual_input])
    else:
        residual = projection
    # highway network
    highway = highway_network(residual, 4)
    # bidirectional gru
    bidirectional_gru = Bidirectional(GRU(CONFIG.embed_size // 2))(highway)
    return bidirectional_gru






