import argparse
import os

import numpy as np
import tensorflow as tf
from keras import backend as K
from scipy.io import wavfile

from keras.layers import Input
from keras.models import Model
from keras.metrics import categorical_accuracy

from config import CONFIG
import components
import data
import audio

# The Storytime model expects a one-hot tensor of character embeddings as input
class StorytimeArchitecture:

    def __init__(self):
        # Placeholders
        self.placeholders()

        # Model input
        self.model_input = self.inputs_placeholder

        # Encoder
        self.encoder_output = components.encoder(self.model_input)

        # Decoder
        go_frame = tf.zeros(shape=tf.shape(self.mel_targets_placeholder))

        decoder_input = K.in_train_phase(self.mel_targets_placeholder, go_frame)
        self.model_output_mel, self.model_output_mag = components.decoder(decoder_input, self.encoder_output)

    def placeholders(self):
        self.inputs_placeholder = K.placeholder(shape=(None, None, CONFIG.embed_size))
        self.mel_targets_placeholder = K.placeholder(shape=(None, None, CONFIG.audio_mel_banks))
        self.mag_targets_placeholder = K.placeholder(shape=(None, None, 1 + (CONFIG.audio_fourier_transform_quantity // 2)))

    def feed_dict(self, inputs_batch, mel_targets_batch, mag_targets_batch, is_training=False):
        feed_dict = {}
        feed_dict[self.inputs_placeholder] = inputs_batch
        feed_dict[self.mel_targets_placeholder] = mel_targets_batch
        feed_dict[self.mag_targets_placeholder] = mag_targets_batch
        feed_dict[K.learning_phase()] = 1 if is_training else 0
        return feed_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', nargs='?', default='./train', type=str)
    parser.add_argument('--eval_path', nargs='?', default='./eval', type=str)

    parser.add_argument('--save_model_file', nargs='?', default='./models/saved_model', type=str)
    parser.add_argument('--generate_sample_every', nargs='?', default=200, type=int)
    args = parser.parse_args()

    train_feature_batches, train_mel_batches, train_mag_batches = data.load_data(args.train_path)
    # eval_feature_batches, eval_target_batches = data.load_data(args.eval_path)

    # Build graph
    architecture = StorytimeArchitecture()

    sess = tf.Session()
    K.set_session(sess)

    mel_loss = tf.reduce_mean(tf.losses.absolute_difference(architecture.mel_targets_placeholder, architecture.model_output_mel))
    mag_loss = tf.reduce_mean(tf.losses.absolute_difference(architecture.mag_targets_placeholder, architecture.model_output_mag))
    total_loss = mel_loss + mag_loss
    optimizer = tf.train.AdamOptimizer(CONFIG.learning_rate).minimize(total_loss)

    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver(tf.trainable_variables())
    #train_writer = tf.summary.FileWriter('./logdir/train', architecture.graph)
    sess.run(init_op)

    with sess.as_default():
        #Train model
        #for i in range(CONFIG.num_epochs):
        i = 0
        minLoss = float("inf")
        while not os.path.isfile("stop_train"):
            i += 1

            totalLoss = 0.0
            lossBatches = 0

            generate_batch = data.generate_batch(train_feature_batches, train_mel_batches, train_mag_batches)
            for features, mel_targets, mag_targets in generate_batch:
                feed_dict = architecture.feed_dict(features, mel_targets, mag_targets, is_training=True)
                #optimizer.run(feed_dict=architecture.feed_dict(features, targets, is_training=True))
                _, loss, mag = sess.run([optimizer, total_loss, architecture.model_output_mag], feed_dict=feed_dict)

                totalLoss += loss
                lossBatches += 1

                # We should generate current output on this iteration
                # This only makes sense if we are training on one example, so we can expect
                # to get that value back
                if args.generate_sample_every != 0 and i % args.generate_sample_every == 0:
                    print("Generating audio...")
                    wavfile.write("./output/output_iter_%d.wav" % (i), \
                                        CONFIG.audio_sample_rate, \
                                        audio.spectrogram2wav(mag))

            averageLoss = totalLoss / lossBatches
            if averageLoss < minLoss:
                print("Writing min loss")
                minLoss = averageLoss
                saver.save(sess, "./models/model_gs_%d_loss_%f" % (i, averageLoss), global_step=i)

            # Save model
            #saver.save(sess, args.save_model_file, global_step=i)
            #train_writer.add_summary(summary, gs)
            print("Epoch %s finished with train loss %.2f" % (i, loss))
