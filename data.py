import numpy as np
import librosa
import re
from keras.preprocessing.text import one_hot
from config import CONFIG

# Generate dummy data
# dummy_train_data = []; dummy_target_data = []
# dummy_num_samples = CONFIG.batch_size * 4
# dummy_vocab_size = CONFIG.embed_size
# dummy_max_length = 50
# for i in range(dummy_num_samples):
#     timesteps = np.random.randint(dummy_max_length, size=1)[0]
#     train_batch = np.random.randint(dummy_vocab_size, size=timesteps)
#     target_batch = np.random.randint(dummy_vocab_size, size=CONFIG.embed_size)
#     dummy_train_data.append(train_batch.tolist())
#     dummy_target_data.append(target_batch.tolist())

def load_texts(dir_str):
	i = 0
	texts = []
	while True:
		file_path = "%s/text/%s.txt" % (dir_str, sub_dir_str, i)
		try:
			file = open(file_path)
			text = file.read()
			text = re.sub(r'[^A-z]+', ' ', text).lower()
			texts.append(list(text))
			i += 1
		except IOError as err: break
	return texts

def load_sounds(dir_str):

# input -> raw text string
# output -> a numpy matrix of (256, characters)
def text_to_sequence(text):
	# split text into a list by character
	# returns the index (row) of where they should appear
	hot_indexes = one_hot(' '.join(list(text)), CONFIG.embed_size, lower=False)

	# placeholder matrix
	matrix = np.zeros(shape=(CONFIG.embed_size, len(hot_indexes)))

	# go through and set the indicated indices "hot" in the create dmatrix
	for i, val in enumerate(hot_indexes):
		matrix[val,i] = 1

	return matrix

def load_data(dir_str):
	texts = load_texts(dir_str)
	sounds = load_sounds(dir_str):
	return (texts, sounds)

# padding must be done here
# keras.preprocessing.sequence.pad_sequences(char_seq, padding='post', value=0)
def generate_batch(train, target, batch_size):
	assert train.shape[0] == target.shape[0]
	l_index = 0
	r_index = batch_size
	while l_index <= train.shape[0]:
		yield (train[l_index:r_index], target[l_index:r_index])
		l_index = r_index
		r_index += batch_size
	yield None
