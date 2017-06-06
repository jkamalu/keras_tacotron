import tensorflow as tf
from keras import backend as K

from keras.layers import Dense, Dropout, Conv1D, BatchNormalization, MaxPooling1D, Bidirectional, Lambda, Reshape, GRU
from keras.layers.merge import Add, Multiply, Concatenate
from keras.initializers import Constant

from config import CONFIG

def decoder(inputs, memory):
    """
    TensorFlow compliant decoder based on MU94W's Tacotron repository.
    See https://github.com/MU94W/Tacotron/blob/master/model.py for source.
    """

    total_steps = CONFIG.max_seq_length
    num_layers = tf.div(total_steps, CONFIG.reduction_factor)
    batch_size = tf.shape(inputs)[0]

    # padding = tf.zeros(shape=(batch_size, CONFIG.max_seq_length - tf.shape(inputs)[1], CONFIG.audio_mel_banks))
    # padded_inputs = Concatenate(axis=1)([inputs, padding])
    # print("padded inputs: %s" % padded_inputs)

    attention_cell = tf.contrib.rnn.GRUCell(CONFIG.embed_size)
    with tf.variable_scope("attention_decoder"):
        attention = tf.contrib.seq2seq.BahdanauAttention(CONFIG.embed_size, memory)
    decoder_gru_1 = tf.contrib.rnn.GRUCell(CONFIG.embed_size)
    decoder_gru_2 = tf.contrib.rnn.ResidualWrapper(tf.contrib.rnn.GRUCell(CONFIG.embed_size))

    with tf.variable_scope("decoder_loop"):
        attention_cell_state = attention_cell.zero_state(batch_size, dtype=tf.float32)
        decoder_gru_1_state = decoder_gru_1.zero_state(batch_size, dtype=tf.float32)
        decoder_gru_2_state = decoder_gru_2.zero_state(batch_size, dtype=tf.float32)
        states = (attention_cell_state, decoder_gru_1_state, decoder_gru_2_state)

        mel_sections = tf.TensorArray(size=total_steps, dtype=tf.float32)

        step = tf.constant(0, dtype=tf.int32)
        condition = lambda step, *vars : tf.less(step, num_layers)

        def decoder_body(step, mel_sections, states):
            l_step = step * CONFIG.reduction_factor
            r_frame = tf.slice(inputs, [0, l_step, 0], [-1, 1, -1])
            r_frame = tf.unstack(r_frame, axis=1)[0]

            with tf.variable_scope('prenet'):
                prenet_output = tf.layers.dense(r_frame, CONFIG.embed_size)
                prenet_output = tf.nn.dropout(prenet_output, CONFIG.dropout)
                prenet_output = tf.layers.dense(prenet_output, CONFIG.embed_size // 2)
                prenet_output = tf.nn.dropout(prenet_output, CONFIG.dropout)

            with tf.variable_scope("attention_cell"):
                attention_cell_output, attention_cell_state = attention_cell(prenet_output, states[0])

            with tf.variable_scope("attention_decoder") as scope:
                context = attention(attention_cell_output)

            with tf.variable_scope("decoder-cells"):
                with tf.variable_scope("cell-1"):
                    decoder_gru_1_output, decoder_gru_1_state = decoder_gru_1(attention_cell_output, states[1]) # context ???
                    decoder_gru_1_res = Add()([tf.identity(attention_cell_output), decoder_gru_1_output])

                with tf.variable_scope("cell-2"):
                    decoder_gru_2_res, decoder_gru_2_state = decoder_gru_2(decoder_gru_1_res, states[2])

            with tf.variable_scope("mel-frames"):
                # mel_frame = Dense(CONFIG.audio_mel_banks)(decoder_gru_2_res)
                mel_frame = tf.layers.dense(decoder_gru_2_res, CONFIG.audio_mel_banks)
                print("mel frame: %s" % mel_frame)
                for r_step in range(CONFIG.reduction_factor):
                    mel_sections = mel_sections.write(l_step + r_step, mel_frame)

            states = attention_cell_state, decoder_gru_1_state, decoder_gru_2_state
            return tf.add(step, 1), mel_sections, states

        __, mel_output, __ = tf.while_loop(condition, decoder_body, (step, mel_sections, states))
        mel_output = mel_output.stack()
        print("mel stack: %s" % mel_output)

    mel_output = tf.reshape(mel_output, [batch_size, total_steps, CONFIG.audio_mel_banks])
    print("mel reshape: %s" % mel_output)
    mag_output = decoder_cbhg(mel_output, mel_output)
    return (mel_output, mag_output)

def decoder_cbhg(inputs, residual_input=None):
    with tf.name_scope('decoder_cbhg'):
        # convolutional bank
        convolutions = convolutional_bank(inputs, decoding=True)
        # max pooling
        max_pooling = MaxPooling1D(pool_size=2, strides=1, padding='same')(convolutions)
        # convolutional projections
        projection = Conv1D(CONFIG.embed_size, 3, padding='same', activation='relu')(max_pooling)
        norm = BatchNormalization()(projection)
        projection = Conv1D(CONFIG.audio_mel_banks, 3, padding='same', activation='linear')(norm)
        norm = BatchNormalization()(projection)
        #residual connection
        if residual_input is not None:
            residual = Add()([norm, residual_input])
        else:
            residual = norm
        # plain feed forward network
        plain = plain_network(residual, num_layers=4)
        # highway network
        highway = highway_network(plain, num_layers=4)
        # bidirectional gru
        bidirectional_gru = Bidirectional(GRU(CONFIG.embed_size // 2, return_sequences=True, implementation=CONFIG.gru_implementation))(highway)
        # linear magnitude spectrogram
        mag_spectrogram = Dense(1 + CONFIG.audio_fourier_transform_quantity // 2)(bidirectional_gru)
        return mag_spectrogram

def encoder(inputs):
    prenet_output = prenet(inputs)
    encoder_output = encoder_cbhg(prenet_output, prenet_output)
    return encoder_output

def prenet(inputs):
    prenet_output = Dense(CONFIG.embed_size, activation='relu')(inputs)
    prenet_output = Dropout(CONFIG.dropout)(prenet_output)
    prenet_output = Dense(CONFIG.embed_size // 2, activation='relu')(prenet_output)
    prenet_output = Dropout(CONFIG.dropout)(prenet_output)
    return prenet_output

def convolutional_bank(inputs, decoding=False):
    convolutions = Conv1D(CONFIG.embed_size // 2, 1, padding='same')(inputs)
    k_width = CONFIG.num_conv_regions if not decoding else CONFIG.num_conv_regions // 2
    for i in range(2, k_width + 1):
        conv = Conv1D(CONFIG.embed_size // 2, i, padding='same')(inputs)
        norm = BatchNormalization()(conv)
        convolutions = Concatenate()([convolutions, conv])
    return convolutions

def plain_network(inputs, num_layers=1):
    for i in range(num_layers):
        inputs = Dense(CONFIG.embed_size // 2, activation='relu')(inputs)
    return inputs

def highway_network(inputs, num_layers=1):
    # https://arxiv.org/pdf/1505.00387.pdf
    # output = H(input,WH) * T(input,WT) + input * C(x,WC)
    for i in range(num_layers):
        H = Dense(CONFIG.embed_size // 2, activation='relu')(inputs)
        T = Dense(CONFIG.embed_size // 2, activation='sigmoid', use_bias=True, bias_initializer=Constant(-1))(inputs)
        C = Lambda(lambda x: 1. - x)(T)
        inputs = Add()([Multiply()([H, T]), Multiply()([inputs, C])])
    return inputs

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
