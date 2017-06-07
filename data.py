import numpy as np
import librosa
import re
from keras.preprocessing.text import one_hot
from config import CONFIG
import audio
import utils
import wave
import collections
import os
import contextlib

# input -> raw text string
# output -> a numpy matrix of (seqlens, 256)
def text_to_sequence(text):
    text = re.sub(r'[^A-z]+', ' ', text)
    # split text into a list by character
    # returns the index (row) of where they should appear
    hot_indexes = one_hot(' '.join(list(text)), CONFIG.embed_size, lower=False)

    # placeholder matrix
    matrix = np.zeros(shape=(len(hot_indexes), CONFIG.embed_size))

    # go through and set the indicated indices "hot" in the create dmatrix
    for i, val in enumerate(hot_indexes):
        matrix[i, val] = 1

    return matrix

# def load_texts(dir_str):
#     i = 0
#     texts = []
#     while True:
#         file_path = "%s/text/%s.txt" % (dir_str, i)
#         try:
#             file = open(file_path)
#             text = file.read()
#         except IOError as err: break
#         texts.append(text_to_sequence(text))
#         i += 1
#     return texts
#
# def load_sounds(dir_str):
#     i = 0
#     mels = []
#     mags = []
#     while True:
#         file_path = "%s/sound/%s.wav" % (dir_str, i)
#         try:
#             mel, mag = audio.get_spectrograms(file_path)
#         except IOError as err: break
#         mels.append(mel)
#         mags.append(mag)
#         i += 1
#     return mels, mags

def make_batch(batch_items, axis_one_dim, spectrogram=False):
    batch_items_padded = []

    # Get max seq len for text batch
    batch_items_max_len = 0
    for matrix in batch_items:
        if matrix.shape[0] > batch_items_max_len:
            batch_items_max_len = matrix.shape[0]

    if spectrogram:
         batch_items_max_len = CONFIG.max_seq_length

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
        batch_mels_padded = make_batch(batch_mels, CONFIG.audio_mel_banks, spectrogram=True)
        mel_batches.append(batch_mels_padded)

        batch_mags = mags[l_index:r_index]
        batch_mags_padded = make_batch(batch_mags, 1 + (CONFIG.audio_fourier_transform_quantity // 2), spectrogram=True)
        mag_batches.append(batch_mags_padded)

        l_index = r_index
        if (r_index + batch_size) < num_samples:
            r_index = r_index + batch_size
        else:
            r_index = num_samples

    return text_batches, mel_batches, mag_batches

def load_data(dir_str, batch_size=CONFIG.batch_size):
    texts = []
    mels = []
    mags = []

    # For housekeeping, keep track of the spectrogram length
    # Bucketed to nearest 100
    spectrogramsLength = collections.defaultdict(int)

    # All possible combinations of books and chapters
    for b in range(1,7+1):
        for ch in range(1,50):
            # Chapter doesn't exist
            if not os.path.isfile("books/book%d/ch%d.txt" % (b, ch)): continue

            chapterSentences = utils.getSentences(b,ch)
            for i, s in enumerate(chapterSentences):
                print("Processing book %d, chapter %d, sentence %d" % (b, ch, i))

                # Sentence audio clip doesn't exist
                audioFilename = "audio/book%d/ch%d/s%d.wav" % (b, ch, i)
                if not os.path.isfile(audioFilename): continue

                text = ' '.join(s)
                mel, mag = audio.get_spectrograms(audioFilename)

                # Make sure the length of the file fits our requirements
                if mel.shape[0] > CONFIG.max_seq_length:
                    print("Skipping (too long) - %s", audioFilename)
                    continue

                texts.append(text_to_sequence(text))
                mels.append(mel)
                mags.append(mag)

    #texts = load_texts(dir_str)
    #mels, mags = load_sounds(dir_str)
    text_batches, mel_batches, mag_batches = make_batches(texts, mels, mags, batch_size)
    return (text_batches, mel_batches, mag_batches)

def generate_batch(train_batches, mel_batches, mag_batches):
    assert len(train_batches) == len(mel_batches)
    assert len(train_batches) == len(mag_batches)
    for i in range(len(train_batches)):
        yield (train_batches[i], mel_batches[i], mag_batches[i])

if __name__ == "__main__":
    train_feature_batches, train_mel_batches, train_mag_batches = load_data('./train', batch_size=1)
    generate_batch = generate_batch(train_feature_batches, train_mel_batches, train_mag_batches)
    for batch in generate_batch:
        for item in batch:
            print(item)
