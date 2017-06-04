# Function should be called:
# python generate.py --file
import argparse
import data
import audio
from config import CONFIG as config
import tensorflow as tf
import numpy as np
from scipy.io import wavfile
from StoryTime import StorytimeArchitecture

def read_text(file_path):
    try:
        file = open(file_path)
        text = file.read()
    except IOError as err:
        text = "empty"
    return text

def generateAudio(text, model, outputFilename):
    modelInput = data.text_to_sequence(text)
    generatedSpectrogram = model.predict(modelInput)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generates a new sound file from the given text')

    parser.add_argument('--model', dest='modelFilename', required=True, help='Saved model file path to use')
    parser.add_argument('--output', dest='outputFilename', required=True, help='Location to save file')
    parser.add_argument('--text', dest='text', required=True, help='Text to generate')

    args = parser.parse_args()

    sess=tf.Session() 
    saver = tf.train.import_meta_graph('%s.meta' % args.modelFilename)
    saver.restore(sess, args.modelFilename)
    sess.run(tf.global_variables_initializer())

    mini_batch = []
    input_seq = data.text_to_sequence(read_text(args.text))
    mini_batch.append(input_seq)
    mini_batch = np.array(mini_batch)

    graph = tf.get_default_graph()
    input_placeholder = graph.get_tensor_by_name("Placeholder:0")
    learning_phase = graph.get_tensor_by_name("dropout_1/keras_learning_phase:0")
    feed_dict ={input_placeholder:mini_batch, learning_phase:False}

    op_to_restore = graph.get_tensor_by_name("decoder_cbhg/dense_22/add:0")
    mag_spectrogram = sess.run(op_to_restore, feed_dict)

    print(mag_spectrogram.shape)

    print("Start spec to wav")

    wavfile.write(args.text, config.audio_sample_rate, audio.spectrogram2wav(mag_spectrogram))



