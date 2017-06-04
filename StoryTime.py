import argparse

import numpy as np
import tensorflow as tf
from keras import backend as K

from keras.layers import Input
from keras.models import Model
from keras.metrics import categorical_accuracy

from config import CONFIG
import components
import data

# The Storytime model expects a one-hot tensor of character embeddings as input
class StorytimeArchitecture:

    def __init__(self):
        self.placeholders()
        self.model_input()
        #self.path = components.encoder_embedding(self.model_input)
        self.path = components.prenet(self.model_input)
        self.path = components.encoder_cbhg(self.path, self.path)
        print("encoder_output: %s" % self.path.get_shape())
        self.model_output()

    def placeholders(self):
        self.inputs_placeholder = K.placeholder(shape=(None, None))
        # targets placeholder shape undecided, dependent on target label representation
        self.targets_placeholder = K.placeholder(shape=(None, None, None))

    def feed_dict(self, inputs_batch, targets_batch, is_training=False):
        feed_dict = {}
        feed_dict[self.inputs_placeholder] = inputs_batch
        feed_dict[self.targets_placeholder] = targets_batch
        feed_dict[K.learning_phase()] = 1 if is_training else 0
        return feed_dict

    def model_input(self):
        self.model_input = self.inputs_placeholder

    def model_output(self):
        self.model_output = self.path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', nargs='?', default='./train', type=str)
    parser.add_argument('--eval_path', nargs='?', default='./eval', type=str)
    parser.add_argument('--save_model_file', nargs='?', default='./saved_model', type=str)
    args = parser.parse_args()

    train_feature_batches, train_target_batches = data.load_data(args.train_path, batch_size=CONFIG.batch_size)
    eval_feature_batches, eval_target_batches = data.load_data(args.eval_path)

    # Build graph
    architecture = StorytimeArchitecture()

    sess = tf.Session()
    K.set_session(sess)

    loss = tf.reduce_mean(tf.losses.absolute_difference(architecture.targets_placeholder, architecture.model_output))
    optimizer = tf.train.AdamOptimizer(CONFIG.learning_rate).minimize(loss)
    acc_value = categorical_accuracy(architecture.targets_placeholder, architecture.model_output)

    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver(tf.trainable_variables())
    sess.run(init_op)

    with sess.as_default():
        #Train model
        for i in range(CONFIG.num_epochs):
            generate_batch = data.generate_batch(train_feature_batches, train_target_batches)
            while True:
                batch = next(generate_batch)
                if batch is None: break
                features, targets = batch
                optimizer.run(feed_dict=architecture.feed_dict(features, targets, is_training=True))
            # Save model
            saver.save(sess, args.save_to_file, global_step=i)

            print("Epoch %s finished" % i)

        # Test model
        evaluation = acc_value.eval(feed_dict=architecture.feed_dict(eval_features, eval_targets))
