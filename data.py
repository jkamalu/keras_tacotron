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

# input -> raw text string
# output -> a numpy matrix of (seqlens, 256)
def text_to_sequence(text):
	# split text into a list by character
	# returns the index (row) of where they should appear
	hot_indexes = one_hot(' '.join(list(text)), CONFIG.embed_size, lower=False)

	# placeholder matrix
	matrix = np.zeros(shape=(len(hot_indexes), CONFIG.embed_size))

	# go through and set the indicated indices "hot" in the create dmatrix
	for i, val in enumerate(hot_indexes):
		matrix[i, val] = 1

	return matrix

def load_texts(dir_str):
	i = 0
	texts = []
	while True:
		file_path = "%s/text/%s.txt" % (dir_str, sub_dir_str, i)
		try:
			file = open(file_path)
			text = file.read()
		except IOError as err: break
		text = re.sub(r'[^A-z]+', ' ', text)
		texts.append(text_to_sequence(text))
		i += 1
	return texts

def load_sounds(dir_str):
	pass

def make_batches(texts, sounds, batch_size):
	num_samples = len(texts)

	text_batches = [] 
	sound_batches = []

	l_index = 0
	r_index = batch_size if batch_size else num_samples

	# make batch to append to batch lists
	while l_index < num_samples:
		batch_texts = texts[l_index:r_index] if r_index < num_samples else texts[l_index:num_samples]
		batch_texts_padded = []

		# Get max seq len for text batch
		batch_texts_max_len = 0
		for matrix in batch_texts:
			if matrix.shape[0] > batch_texts_max_len:
				batch_texts_max_len = matrix.shape[0]

		# Pad text sequences and create 3d numpy array
		for matrix in batch_texts:
			padding = np.zeros(shape=(batch_texts_max_len - matrix.shape[0], 256))
			padded_matrix = np.append(matrix, padding, axis=0)
			batch_texts_padded.append(padded_matrix)

		# Append text batch to text batch list
		text_batches.append(np.array(batch_texts_padded))

		l_index = r_index
		r_index = r_index + batch_size

	return text_batches, sound_batches

def load_data(dir_str, batch_size=None):
	texts = load_texts(dir_str)
	sounds = load_sounds(dir_str)
	return make_batches(texts, sounds, batch_size)

def generate_batch(train_batches, target_batches):
	assert len(train_batches) == len(target_batches)
	for i in range(len(train_batches)):
		yield (train_batches[i], target_batches[i])




