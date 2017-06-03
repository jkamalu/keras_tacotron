from scipy.io import wavfile

import numpy as np
import copy
import librosa
from config import CONFIG as config

def get_spectrograms(sound_file):
    '''Extracts melspectrogram and magnitude from given `sound_file`.
    Args:
      sound_file: A string. Full path of a sound file.
    Returns:
      Transposed S: A 2d array. A transposed melspectrogram with shape of (T, n_mels)
      Transposed magnitude: A 2d array. A transposed magnitude spectrogram
        with shape of (T, 1+hp.n_fft//2)
    '''
    # Loading sound file
    y, sr = librosa.load(sound_file, sr=None) # or set sr to hp.sr.

    if sr != config.audio_sample_rate:
        print "Sample rates should be the same (%d vs %d)" % (sr, config.audio_sample_rate)
        print "Adjusting now..."
        y = librosa.core.resample(y, sr, config.audio_sample_rate)
        sr = config.audio_sample_rate

    # stft. D: (1+n_fft//2, T)
    D = librosa.stft(y=y,
                     n_fft=config.audio_fourier_transform_quantity,
                     hop_length=config.audio_hop_length,
                     win_length=config.audio_window_length)

    # power magnitude spectrogram
    magnitude = np.abs(D)**config.audio_mel_magnitude_exp #(1+n_fft/2, T)

    # mel spectrogram
    S = librosa.feature.melspectrogram(S=magnitude, n_mels=config.audio_mel_banks) #(n_mels, T)

    return np.transpose(S.astype(np.float32)), np.transpose(magnitude.astype(np.float32)) # (T, n_mels), (T, 1+n_fft/2)

def invert_spectrogram(spectrogram):
    return librosa.istft(spectrogram.T, \
                        hop_length=config.audio_hop_length, \
                        win_length=config.audio_window_length, \
                        window="hann")

def spectrogram2wav(spectrogram):
    X_best = copy.deepcopy(spectrogram)
    for i in range(config.audio_inversion_iterations):
        X_t = invert_spectrogram(X_best)
        est = librosa.stft(X_t, config.audio_fourier_transform_quantity, config.audio_hop_length).T
        phase = est / np.maximum(1e-8, np.abs(est))
        X_best = spectrogram * phase[:len(spectrogram)]
    X_t = invert_spectrogram(X_best)

    return np.real(X_t)

melSpectrogram, linearSpectrogram = get_spectrograms("example.wav")
wavfile.write("example_output.wav", config.audio_sample_rate, spectrogram2wav(linearSpectrogram))
