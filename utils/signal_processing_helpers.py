import librosa
from librosa.util import normalize
import numpy as np
from scipy.signal import butter, lfilter


def add_noise(data, noise_ratio=.05):
    """
    adds randomness (white noise) to signal
    Args:
        data: array of audio file(s)
        noise_ratio: how much noise to add

    Returns: normalized audio file with white noise applied

    """
    noisy_data = data + noise_ratio * np.random.normal(loc=0.0, scale=1.0, size=data.shape)
    return normalize(noisy_data)

def butter_bandpass(lowcut, highcut, fs, order=5):
    """
    creates butter bandpass filter to constrain frequency range
    Args:
        lowcut: low frequency cutoff point
        highcut: high frequency cutoff point
        fs: sample rate
        order: roll-off (smaller is more aggressive)

    Returns: Numerator (b) and denominator (a) polynomials of the IIR filter

    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5, normalize=False):
    """
    applies butter bandpass to audio array
    Args:
        data: 1D array audio file
        lowcut: low frequency cutoff point
        highcut: high frequency cutoff point
        fs: sample rate
        order: roll-off (smaller is more aggressive)
        normalize: if True, normalizes data

    Returns: 1D array audio file with butter bandpass filter applied

    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    if normalize:
        y = normalize(y)
    return y


def make_log_mel_spectrogram(audio_array, sr, n_mels=64):
    """

    Args:
        audio_array: 1D array of audio data
        sr: sample rate of audio_array
        n_mels: number of mel bins

    Returns: log_mel_spectrogram

    """
    S = librosa.feature.melspectrogram(y=audio_array, sr=sr, n_mels=n_mels)
    S = librosa.power_to_db(S, ref=np.max)
    return S