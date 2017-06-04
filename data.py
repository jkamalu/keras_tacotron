import numpy as np
import librosa
import re
from keras.preprocessing.text import one_hot
from config import CONFIG
import audio

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
        file_path = "%s/text/%s.txt" % (dir_str, i)
        try:
            file = open(file_path)
            text = file.read()
        except IOError as err: break
        text = re.sub(r'[^A-z]+', ' ', text)
        texts.append(text_to_sequence(text))
        i += 1
    return texts

def load_sounds(dir_str):
    i = 0
    mels = []
    mags = []
    while True:
        file_path = "%s/sound/%s.wav" % (dir_str, i)
        try:
            mel, mag = audio.get_spectrograms(file_path)
        except IOError as err: break
        mels.append(mel)
        mags.append(mag)
        i += 1
    return mels, mags

def make_batch(batch_items, axis_one_dim):
    batch_items_padded = []

    # Get max seq len for text batch
    batch_items_max_len = 0
    for matrix in batch_items:
        if matrix.shape[0] > batch_items_max_len:
            batch_items_max_len = matrix.shape[0]

    # Pad text sequences and create 3d numpy array
    for matrix in batch_items:
        padding = np.zeros(shape=(batch_items_max_len - matrix.shape[0], axis_one_dim))
        padded_matrix = np.append(matrix, padding, axis=0)
        batch_items_padded.append(padded_matrix)

    return np.array(batch_items_padded)

def make_batches(texts, mels, mags, batch_size):
    num_samples = len(texts)

    text_batches = [] 
    mel_batches = []
    mag_batches = []

    l_index = 0
    if batch_size < num_samples:
        r_index = batch_size
    else:
        r_index = num_samples

    # make batch to append to batch lists
    while l_index < num_samples:
        batch_texts = texts[l_index:r_index]
        batch_texts_padded = make_batch(batch_texts, CONFIG.embed_size)
        text_batches.append(batch_texts_padded)

        batch_mels = mels[l_index:r_index]
        batch_mels_padded = make_batch(batch_mels, CONFIG.audio_mel_banks)
        mel_batches.append(batch_mels_padded)

        batch_mags = mags[l_index:r_index]
        batch_mags_padded = make_batch(batch_mags, 1 + (CONFIG.audio_fourier_transform_quantity // 2))
        mag_batches.append(batch_mags_padded)       

        l_index = r_index
        if (r_index + batch_size) < num_samples:
            r_index = r_index + batch_size
        else:
            r_index = num_samples

    return text_batches, mel_batches, mag_batches

def load_data(dir_str, batch_size=32):
    texts = load_texts(dir_str)
    mels, mags = load_sounds(dir_str)
    text_batches, mel_batches, mag_batches = make_batches(texts, mels, mags, batch_size)
    return (text_batches, mel_batches, mag_batches)

def generate_batch(train_batches, mel_batches, mag_batches):
    assert len(train_batches) == len(mel_batches)
    assert len(train_batches) == len(mag_batches)
    for i in range(len(train_batches)):
        yield (train_batches[i], mel_batches[i], mag_batches[i])

if __name__ == "__main__":
    for elem in load_data('./train')[2]:
        print("boop")
        print(elem)