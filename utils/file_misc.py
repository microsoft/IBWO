import csv
import os

import librosa

def get_wav_files(base_path):
    """

    Args:
        base_path: path that contains .wav files

    Returns: list of full file paths to .wav files

    """
    _all_files = []
    for r, d, f in os.walk(base_path):
        for file in f:
            if file.lower().endswith('.wav'):
                _all_files.append(os.path.join(r, file))
    return _all_files


def load_all_files(list_of_files, sample_rate):
    """
    Args:
        list_of_files: list of paths to .wav files, as returned by bet_wav_files
        sample_rate: sample rate of audio, select None to keep sample rate of .wav file

    Returns: dict of {path: audio_array}

    """
    file_dict = {}
    for file in list_of_files:
        try:
            y, sr = librosa.load(file, sr=sample_rate)
            file_dict[file] = y
        except:
            print('problem_with_file:', file)
    return file_dict

def write_csv_row(row_list, filename, overwrite_file=False):
    """
    writes .csv file to keep track of progress
    Args:
        row_list: a list of items to write to a row
        filename: csv filename (including path)
        overwrite_file: True to overwrite existing file, False to append to existing

    Returns: None

    """
    if overwrite_file:
        write = 'w'
    else:
        write = 'a'
    with open(filename, write, newline='') as writer:
        writer = csv.writer(writer)
        writer.writerow(row_list)