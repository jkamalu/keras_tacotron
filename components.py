import tensorflow as tf
from keras import backend as K

from keras.layers import Dense, Dropout, Conv1D, BatchNormalization, MaxPooling1D, GRU, Bidirectional, Lambda
from keras.layers.merge import Add, Multiply, Concatenate
from keras.initializers import Constant

from config import CONFIG

def encoder_embedding(inputs):
    with tf.name_scope('encoder_embedding'):
        embedding = Lambda(lambda x: tf.one_hot(tf.to_int32(x), depth=CONFIG.embed_size))(inputs)
        return embedding

def seq_decoder(inputs, memory):
    prenet_inputs = prenet(inputs)

    dec = attention(prenet_inputs, memory, variable_scope="attention_decoder1")      #Might want to separate into variable scopes
    dec = Add()([dec, attention(dec, memory, variable_scope="attention_decoder2")])

    #we need to capture reduction_factor frames with mel_bank features each, hence the multiplied dimensionality
    out_dim = CONFIG.audio_mel_banks*CONFIG.reduction_factor

    #original uses a fully connected layer, whereas we use a dense layer
    #If this presents problems take a look at simpleRNN
    outputs = Dense(out_dim)(dec)
    return outputs

def decoder_cbhg(inputs):
    with tf.name_scope('decoder_cbhg'):
        # convolutional bank
        convolutions = convolutional_bank(inputs, True)
        # max pooling
        max_pooling = MaxPooling1D(pool_size=2, strides=1, padding='same')(convolutions)
        # convolutional projections
        projection = Conv1D(CONFIG.embed_size, 3, padding='same', activation='relu')(max_pooling)
        norm = BatchNormalization()(projection)
        projection = Conv1D(CONFIG.num_mel_bands, 3, padding='same', activation='linear')(norm)
        norm = BatchNormalization()(projection)

        #residual connections
        if residual_input is not None:
            residual = Add()([norm, residual_input])
        else:
            residual = norm
        # highway network
        highway = highway_network(residual, num_layers=4)
        # bidirectional gru
        bidirectional_gru = Bidirectional(GRU(CONFIG.embed_size // 2, return_sequences=True))(highway)
        return bidirectional_gru

def attention(inputs, memory, num_units=256, variable_scope="attention_decoder"):
    # Tensorflow underlying code to support Bahdanau attention
    # Returns a tensorflow
    def attend(inp, mem):
        with tf.variable_scope(variable_scope):
            # The attention component will be in control of attending to the given memory
            attention = tf.contrib.seq2seq.BahdanauAttention(num_units, mem)
            cell = tf.contrib.rnn.GRUCell(num_units)

            cell_with_attention = tf.contrib.seq2seq.DynamicAttentionWrapper(cell, attention, num_units)
            outputs, _ = tf.nn.dynamic_rnn(cell_with_attention, inp, dtype=tf.float32)
            return outputs

    # output should be [batches, timesteps, num_units]
    #return Lambda(attend, output_shape=(b,t,num_units))(inputs, memory)
    return Lambda(attend)(inputs, memory)

def prenet(inputs):
    prenet_output = Dense(CONFIG.embed_size, activation='relu')(inputs)
    prenet_output = Dropout(0.5)(prenet_output)
    prenet_output = Dense(CONFIG.embed_size // 2, activation='relu')(prenet_output)
    prenet_output = Dropout(0.5)(prenet_output)
    return prenet_output

def convolutional_bank(inputs, decoding=False):
    convolutions = Conv1D(CONFIG.embed_size // 2, 1, padding='same')(inputs)

    k_width = CONFIG.num_conv_regions if decoding == False else CONFIG.num_conv_regions//2
    for i in range(2, k_width + 1):
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
    with tf.name_scope('encoder_cbhg'):
        # convolutional bank
        convolutions = convolutional_bank(inputs)
        # max pooling
        max_pooling = MaxPooling1D(pool_size=2, strides=1, padding='same')(convolutions)
        # convolutional projections
        projection = Conv1D(CONFIG.embed_size // 2, 3, padding='same', activation='relu')(max_pooling)
        norm = BatchNormalization()(projection)
        projection = Conv1D(CONFIG.embed_size // 2, 3, padding='same', activation='linear')(norm)
        norm = BatchNormalization()(projection)
        # residual connection
        if residual_input is not None:
            residual = Add()([norm, residual_input])
        else:
            residual = norm
        # highway network
        highway = highway_network(residual, num_layers=4)
        # bidirectional gru
        bidirectional_gru = Bidirectional(GRU(CONFIG.embed_size // 2, return_sequences=True))(highway)
        return bidirectional_gru
