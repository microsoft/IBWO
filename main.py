import os

import librosa
import numpy as np
from scipy.stats import pearsonr
import torch

from cnn_model.make_train_data_pitt import get_train_arrays
from cnn_model.train import AudioCNN
# from utils.azure_blob_helpers import get_blobs_from_dir, download_blob, block_blob_service
from utils.file_misc import write_csv_row, get_wav_files, load_all_files
from utils.signal_processing_helpers import make_log_mel_spectrogram


def match_segments(source_file_dict, field_recording, source_base_path=''):
    results_dict = {'best_p':1, 'best_r':0, 'best_idx': None, 'best_source_file': ''}
    y = field_recording
    for file, y2 in source_file_dict.items():
        filename = file.strip(source_base_path)
        start = 0
        while start <= len(y) - len(y2):
            temp_window = y[start:start + len(y2)]
            r, p = pearsonr(temp_window, y2)
            if abs(r) > abs(results_dict['best_r']):
                results_dict['best_r'] = r
                results_dict['best_idx'] = start
                results_dict['best_source_file'] = filename
            if p < results_dict['best_r']:
                results_dict['best_p'] = p
            start += int(len(y2))
    return results_dict

def get_model_score(model, field_array_dict, epsilon=1e-6):
    all_arrays = get_train_arrays(field_array_dict)
    results_dict = {'max_pos_score': 0, 'best_second_markers': []}
    all_S = []
    for i, a in enumerate(all_arrays):
        S = make_log_mel_spectrogram(a, sr=32000)
        all_S.append(np.expand_dims(S, axis=0))
    spectrogram = np.array(all_S)
    spectrogram = spectrogram.astype(np.float32)
    spectrogram = torch.from_numpy(spectrogram)
    in_data = spectrogram.to(device)
    results = model(in_data)
    all_true = []
    # for r in results:
    #     all_true.append(r.cpu().detach().numpy()[0])
    # all_true = np.array(all_true)
    # results_dict['max_pos_score'] = all_true.max()
    # results_dict['best_second_markers'] = list(np.where(all_true == all_true.max())[0])
    for i, a in enumerate(all_arrays):
        spectrogram = make_log_mel_spectrogram(a, 32000)
        spectrogram = np.expand_dims(spectrogram, axis=0)
        spectrogram = np.expand_dims(spectrogram, axis=0)
        spectrogram = spectrogram.astype(np.float32)
        spectrogram = torch.from_numpy(spectrogram)
        in_data = spectrogram.to(device)
        with torch.no_grad():
            results = model(in_data)
            pos_result = results.cpu().detach().numpy()[0][0]
            if pos_result > results_dict['max_pos_score']:
                results_dict['max_pos_score'] = pos_result
                results_dict['best_second_markers'] = [i]
            elif abs(pos_result - results_dict['max_pos_score']) < epsilon:
                results_dict['best_second_markers'].append(i)
    return results_dict

def analyze_file(path_to_field_file, csv_save_path, file_base_path=''):
    results_dict = {k:'' for k in csv_header}
    results_dict['field_recording_file'] = path_to_field_file.strip(file_base_path)
    if os.path.getsize(path_to_field_file) < 1000:
        return None
    y, sr = librosa.load(path_to_field_file, sr=None)

    results_1935 = match_segments(IBWO_1935_resampled, y, source_base_path=IBWO_1935_path)
    results_dict['best_r_1935'] = results_1935['best_r']
    results_dict['best_p_1935'] = results_1935['best_p']
    results_dict['best_r_index_1935'] = results_1935['best_idx']
    results_dict['best_1935_file'] = results_1935['best_source_file']
    
    results_aru = match_segments(IBWO_aru_resampled, y, source_base_path=IBWO_aru_path)
    results_dict['best_r_aru'] = results_aru['best_r']
    results_dict['best_p_aru'] = results_aru['best_p']
    results_dict['best_r_index_aru'] = results_aru['best_idx']
    results_dict['best_aru_file'] = results_aru['best_source_file']

    model_results = get_model_score(model, {'':y})
    results_dict['best_model_score'] = model_results['max_pos_score']
    results_dict['best_model_score_second_marking(s)'] = model_results['best_second_markers']

    csv_row = [results_dict[k] for k in csv_header]
    write_csv_row(csv_row, csv_save_path, overwrite_file=False)


def write_csv_file(field_recording_folder, header_list):
    save_path, filename = os.path.split(field_recording_folder)
    print('filename', filename)
    if filename.endswith('/'):
      filename = filename[0:-1]
    full_csv_path = os.path.join(save_path, filename + '.csv')
    print('csv path', full_csv_path)
    write_csv_row(header_list, full_csv_path, overwrite_file=True)
    for r, d, f in os.walk(field_recording_folder):
        for sound_file in f:
            if sound_file.lower().endswith('.wav'):

                path_to_field_recording = os.path.join(r, sound_file)
                print(path_to_field_recording)
                analyze_file(path_to_field_recording,
                             full_csv_path,
                             file_base_path=save_path)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Process a folder of field recordings')
    parser.add_argument('input_dir', type=str, help='a folder that contains audio files')

    args = parser.parse_args()
    input_dir = args.input_dir
    if input_dir.endswith('/'):
      input_dir = input_dir[0:-1]
    csv_header = ['field_recording_file', 'best_model_score', 'best_model_score_second_marking(s)',
                  'best_r_1935', 'best_p_1935', 'best_r_index_1935', 'best_1935_file',
                  'best_r_aru', 'best_p_aru', 'best_r_index_aru', 'best_aru_file']
    # csv_header = ['field_recording_file', 'best_model_score', 'second_marking(s)',
    #               'best_r_1935', 'best_p_1935', 'best_r_index_1935', 'best_1935_file']
    # csv_header = ['field_recording_file', 'best_model_score', 'second_marking(s)']    # csv_save_path = 'first_run.csv'
    # write_csv_row(csv_header, csv_save_path, overwrite_file=True)
    IBWO_1935_path = "/home/ibwo/IBWO-kent templates/"
    IBWO_1935_files = get_wav_files(IBWO_1935_path)
    IBWO_1935_resampled = load_all_files(IBWO_1935_files, sample_rate=32000)
    #
    IBWO_aru_path = "/home/ibwo/IBWO-aru/aru-recordings"
    IBWO_aru_files = get_wav_files(IBWO_aru_path)
    IBWO_aru_resampled = load_all_files(IBWO_aru_files, sample_rate=32000)

    # model_parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load('cnn_model/models/20190809_173904_val_loss=0.0085.ckpt', map_location=torch.device('cpu')) #"cnn_model/models/20190723_162437_val_loss=0.0000.ckpt"
    model = AudioCNN()
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    write_csv_file(args.input_dir, csv_header)

