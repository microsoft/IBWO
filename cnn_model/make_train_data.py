import os

import librosa
import numpy as np
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split

from utils.file_misc import get_wav_files, load_all_files
from utils.signal_processing_helpers import make_log_mel_spectrogram

IBWO_1935_path = "" #path to 1935 recordings
IBWO_1935_files = get_wav_files(IBWO_1935_path)
IBWO_1935_resampled = load_all_files(IBWO_1935_files, sample_rate=32000)

ARU_path = "" #path to ARU recordings or others
ARU_files = get_wav_files(ARU_path)
ARU_resampled = load_all_files(ARU_files, sample_rate=32000)


def get_train_arrays(data_dict):
    all_train_arrays = []
    for y in data_dict.values():
    #     print(f)
        if len(y) > 32000:
            start = 0
            end = start + 32000
            while start + 32000 <= len(y):
                all_train_arrays.append(y[start:end])
                start += 32000
                end += 32000
            if end - 32000 + 1 < len(y):
                all_train_arrays.append(np.concatenate((y[end:], np.zeros(32000 - len(y[end:])))))
        else:
            all_train_arrays.append(np.concatenate((y, np.zeros(32000 - len(y)))))
    return np.array(all_train_arrays)


if __name__ == '__main__':
    ## get_true_labels
    ibwo_1935 = get_train_arrays(IBWO_1935_resampled)
    aru = get_train_arrays(ARU_resampled)
    
    X_train_true = ibwo_1935
    y_train_true = np.array([[1, 0] for i in range(len(X_train_true))])

    X_test_true = aru
    y_test_true = np.array([[1, 0] for i in range(len(X_test_true))])

    ## get_false_labels
    no_woodpecker_files_train = "" #path to folder of other sounds
    input_dict_train = {}
    for file in os.listdir(no_woodpecker_files_train):
        file = os.path.join(no_woodpecker_files_train, file)
        y, sr = librosa.load(file, sr=32000)
        input_dict_train[file] = y
    X_train_false = get_train_arrays(input_dict_train)
    y_train_false = np.array([[0, 1] for i in range(len(X_train_false))])

    no_woodpecker_files_test = "" #path to folder of other sounds
    input_dict_test = {}
    for file in os.listdir(no_woodpecker_files_test):
        file = os.path.join(no_woodpecker_files_test, file)
        y, sr = librosa.load(file, sr=32000)
        input_dict_test[file] = y
    X_test_false = get_train_arrays(input_dict_test)
    y_test_false = np.array([[0, 1] for i in range(len(X_test_false))])

    X_train = np.concatenate((X_train_true, X_train_false))
    X_test = np.concatenate((X_test_true, X_test_false))
    y_train = np.concatenate((y_train_true, y_train_false))
    y_test = np.concatenate((y_test_true, y_test_false))

    X_train = np.array([make_log_mel_spectrogram(y, sr=32000) for y in X_train])
    X_test = np.array([make_log_mel_spectrogram(y, sr=32000) for y in X_test])
    # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, shuffle=True)
    for i, (x, y) in enumerate(zip(X_train, y_train)):
        joblib.dump((x, y), 'train_files/{}.pkl'.format(i))

    for i, (x, y) in enumerate(zip(X_test, y_test)):
        joblib.dump((x, y), 'test_files/{}.pkl'.format(i))
