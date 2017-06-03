import numpy as np

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

# input  -> dataset directory
# output -> (padded character sequences, spectrogram targets) pair of numpy ndarrays
def load_data(directory):
	# padding example
	# keras.preprocessing.sequence.pad_sequences(char_seq, padding='post', value=0)
	return (None, None)


def generate_batch(train, target, batch_size):
	assert train.shape[0] == target.shape[0]
	l_index = 0
	r_index = batch_size
	while l_index <= train.shape[0]:
		yield (train[l_index:r_index], target[l_index:r_index])
		l_index = r_index
		r_index += batch_size
	yield None
