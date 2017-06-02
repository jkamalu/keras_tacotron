import numpy as np
import tensorflow as tf
from keras import backend as K

from keras.layers import Input
from keras.models import Model
from keras.preprocessing import sequence
from keras.metrics import categorical_accuracy

from config import CONFIG
import components
import data

class StorytimeArchitecture:

    def __init__(self):
        self.placeholders()
        self.model_input()
        self.path = components.encoder_embedding(self.model_input)
        self.path = components.encoder_prenet(self.path)
        self.path = components.encoder_cbhg(self.path, self.path)
        print("encoder_output: %s" % self.path.get_shape())
        self.model_output()

    def placeholders(self):
        self.inputs_placeholder = K.placeholder(shape=(None, None))
        self.targets_placeholder = K.placeholder(shape=(None, None))

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
    # Generate dummy data
    dummy_train_data = []; dummy_target_data = []
    dummy_num_samples = CONFIG.batch_size * 4
    dummy_vocab_size = CONFIG.embed_size
    dummy_max_length = 50
    for i in range(dummy_num_samples):
        timesteps = np.random.randint(dummy_max_length, size=1)[0]
        train_batch = np.random.randint(dummy_vocab_size, size=timesteps)
        target_batch = np.random.randint(dummy_vocab_size, size=CONFIG.embed_size)
        dummy_train_data.append(train_batch.tolist())
        dummy_target_data.append(target_batch.tolist())

    # Must be accounted for after data has been loaded
    train_data = dummy_train_data
    target_data = dummy_target_data
    num_samples = dummy_num_samples

    # Pad inputs
    train_data = sequence.pad_sequences(train_data, padding='post', value=0)
    target_data = sequence.pad_sequences(target_data, padding='post', value=0)

    # Build graph
    architecture = StorytimeArchitecture()

    sess = tf.Session()
    K.set_session(sess)

    loss = tf.reduce_mean(tf.losses.absolute_difference(architecture.targets_placeholder, architecture.model_output))
    optimizer = tf.train.AdamOptimizer(CONFIG.learning_rate).minimize(loss)

    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    generate_batch = data.generate_batch(train_data, target_data, CONFIG.batch_size)
    batches_per_epoch = math.ceil(num_samples) 

    with sess.as_default():
        while True:
            batch = next(generate_batch)
            if batch is None: break
            train, target = batch
            optimizer.run(feed_dict=architecture.feed_dict(train, target, is_training=True))

    acc_value = categorical_accuracy(architecture.targets_placeholder, architecture.model_output)
    with sess.as_default():
        evaluation = acc_value.eval(feed_dict=architecture.feed_dict(train, target))










