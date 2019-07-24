import os

import numpy as np
from sklearn.externals import joblib
import torch
from torch.utils.data.dataset import Dataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AudioDataset(Dataset):

    def __init__(self, data_path):
        """

        Args:
            data_path: path to .pkl files, each of which is a tuple of (spectrogram, label)
        """
        if not os.path.exists(data_path):
            raise Exception('data path does not exist')
        self.data_path = [os.path.join(data_path, f) for f in os.listdir(data_path)]
        self.data_len = len(self.data_path)

    def __getitem__(self, index):

        self.filename = os.path.basename(self.data_path[index])

        spectrogram, label = joblib.load(self.data_path[index])
        spectrogram = np.expand_dims(spectrogram, 0)
        spectrogram = spectrogram.astype(np.float32)
        spectrogram = torch.from_numpy(spectrogram)
        spectrogram = spectrogram.to(device)
        label = label.astype(np.float32)
        label = torch.from_numpy(label)
        label = label.to(device)
        return spectrogram, label


    def __len__(self):
        return self.data_len
