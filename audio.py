from scipy.io import wavfile
from scipy.signal import spectrogram
import numpy as np
import copy
import librosa
from config import CONFIG as config

# Returns {"mel": XX, "linear": XX} to signify the two types of spectrograms that
# are required in this model
def get_spectrograms(filename):
    audio, sampleRate = librosa.load(filename, sr=None)
    print audio

    # Goal 1: Retrieve the magnitude of the standard spectrogram
    magnitude1 = spectrogram(audio, \
                            fs=sampleRate, \
                            nperseg=config.frame_length, \
                            noverlap=config.frame_length-config.frame_shift, \
                            window=config.window_type, \
                            nfft=config.fourier_transform_quantity, \
                            mode="magnitude")
    print magnitude1

    audio, sampleRate = librosa.load(filename, sr=None)

    D = librosa.stft(y=audio,
                     n_fft=2048,
                     hop_length=0.0125,
                     win_length=0.050)
    #magnitude = np.abs(D)**hp.power #(1+n_fft/2, T)
    magnitude = np.abs(D)
    print magnitude


melSpectrogram, linearSpectrogram = get_spectrograms("example.wav")


def get_spectrograms(sound_file):
    global sr
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

    # stft. D: (1+n_fft//2, T)
    D = librosa.stft(y=y,
                     n_fft=hp.n_fft,
                     hop_length=hp.hop_length,
                     win_length=hp.win_length)

    # power magnitude spectrogram
    magnitude = np.abs(D)**hp.power #(1+n_fft/2, T)

    # mel spectrogram
    S = librosa.feature.melspectrogram(S=magnitude, n_mels=hp.n_mels) #(n_mels, T)

    return np.transpose(S.astype(np.float32)), np.transpose(magnitude.astype(np.float32)) # (T, n_mels), (T, 1+n_fft/2)

def invert_spectrogram(spectrogram):
    return librosa.istft(spectrogram.T, hp.hop_length, win_length=hp.win_length, window="hann")

def spectrogram2wav(spectrogram):
    X_best = copy.deepcopy(spectrogram)
    for i in range(hp.n_iter):
        X_t = invert_spectrogram(X_best)
        est = librosa.stft(X_t, hp.n_fft, hp.hop_length).T
        phase = est / np.maximum(1e-8, np.abs(est))
        X_best = spectrogram * phase[:len(spectrogram)]
    X_t = invert_spectrogram(X_best)

    return np.real(X_t)

#melSpectrogram, linearSpectrogram = get_spectrograms("example.wav")
#wavfile.write("example_output.wav", sr, spectrogram2wav(linearSpectrogram))
