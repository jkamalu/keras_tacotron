import tensorflow as tf
from keras import backend as K

from keras.layers import Dense, Dropout, Conv1D, BatchNormalization, MaxPooling1D, GRU, Bidirectional, Lambda, Reshape
from keras.layers.merge import Add, Multiply, Concatenate
from keras.initializers import Constant

from config import CONFIG

def seq_decoder(inputs, memory, scope="decoder1", reuse=None):
    #going off the assumption that we have to run the decoder for r number of steps, feeding the r frames every time

    #for i in range(0, CONFIG.reduction_factor):
    with tf.variable_scope(scope, reuse=reuse):
        prenet_inputs = prenet(inputs)

        dec = attention(prenet_inputs, memory, reuse=True)

        #By maintaining the tf variable scope within the two layers it should allow for weights to be passed horizontally
        #Else we might need to implement our own GRU or use a pure TF one
        with tf.variable_scope("decoder_rnn1"):
            #just uses attention as input rather than residual, which seems more faithful to the paper than other
            #implementations, old form is:
                #dec = Add()([dec, GRU(CONFIG.embed_size, return_sequences=True, implementation=CONFIG.gru_implementation)(dec)])
            dec = GRU(CONFIG.embed_size, return_sequences=True, implementation=CONFIG.gru_implementation)(dec)
            dec = Add()([dec, GRU(CONFIG.embed_size, return_sequences=True, implementation=CONFIG.gru_implementation)(dec)])

        #otputs = Dense(out_dim)(dec)
        inputs = dec

    #we need to capture reduction_factor frames with mel_bank features each, hence the multiplied dimensionality
    out_dim = CONFIG.audio_mel_banks*CONFIG.reduction_factor
    outputs = Dense(out_dim)(inputs)
    #_, output = tf.split(outputs, [out_dim-CONFIG.audio_mel_banks, CONFIG.audio_mel_banks], axis=-1)
    outputs = tf.split(outputs, num_or_size_splits=5, axis=2)
    #print(outputs)
    return outputs[-1]

def decoder_cbhg(inputs, residual_input=None):
    with tf.name_scope('decoder_cbhg'):
        # convolutional bank
        convolutions = convolutional_bank(inputs, True)
        # max pooling
        max_pooling = MaxPooling1D(pool_size=2, strides=1, padding='same')(convolutions)

        # convolutional projections
        projection = Conv1D(CONFIG.embed_size, 3, padding='same', activation='relu')(max_pooling)
        norm = BatchNormalization()(projection)
        projection = Conv1D(CONFIG.embed_size//2, 3, padding='same', activation='linear')(norm)
        norm = BatchNormalization()(projection)

        #residual connections
        if residual_input is not None:
            residual = Add()([norm, residual_input])
        else:
            residual = norm

        #if we want to have 80-dim conv converted to 128 instead of a 128 conv with no conversion (current)
        #projection = Conv1D(CONFIG.audio_mel_banks, 3, padding='same', activation='linear')(norm)
        #res_dim = CONFIG.embed_size//2
        #residual = Dense(res_dim)(residual)

        # highway network
        highway = highway_network(residual, num_layers=4)

        # bidirectional gru
        bidirectional_gru = Bidirectional(GRU(CONFIG.embed_size // 2, return_sequences=True, implementation=CONFIG.gru_implementation))(highway)

        out_dim = (1 + CONFIG.audio_fourier_transform_quantity//2)
        outputs = Dense(out_dim)(bidirectional_gru)
        return outputs

# Tensorflow underlying code to support Bahdanau attention
def attention(inputs, memory, num_units=CONFIG.embed_size, variable_scope="attention_decoder", reuse=None):
    
    def attend(input_and_memory):
        inputs, memory = input_and_memory
        with tf.variable_scope(variable_scope) as scope:
            try:
                v = tf.get_variable("v", [1])
            except ValueError:
                scope.reuse_variables()
            # The attention component will be in control of attending to the given memory
            attention = tf.contrib.seq2seq.BahdanauAttention(num_units, memory)
            #reuse=True
            cell = tf.contrib.rnn.GRUCell(num_units)

            cell_with_attention = tf.contrib.seq2seq.DynamicAttentionWrapper(cell, attention, num_units)
            #second output is the state, paper mentions we want a stateful recurrent
            #layer to produce the attn query at each decoder timestep, so might need to use it
            outputs, _ = tf.nn.dynamic_rnn(cell_with_attention, inputs, dtype=tf.float32)
            return outputs

    # output should be [batches, timesteps, num_units]
    #return Lambda(attend, output_shape=(b,t,num_units))(inputs, memory)
    return Lambda(attend)([inputs, memory])

def encoder(inputs):
    prenet_output = prenet(inputs)
    encoder_output = encoder_cbhg(prenet_output, prenet_output)
    return encoder_output

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
    bidirectional_gru = Bidirectional(GRU(CONFIG.embed_size // 2, return_sequences=True, implementation=CONFIG.gru_implementation))(highway)
    return bidirectional_gru
