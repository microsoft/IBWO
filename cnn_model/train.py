"""
trains model, saves checkpoint files of best model
based on: https://github.com/microsoft/dcase-2019
"""

from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from cnn_model.load_dataset import AudioDataset

DATE = datetime.now().strftime('%Y%m%d_%H%M%S')

RUN_NAME = DATE

BATCH_SIZE = 15
NUM_CLASSES = 2
MAX_NUM_EPOCHS = 10000

## load model from checkpoint
CHECKPOINT = False
CHECKPOINT_PATH = ''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConvBlock(nn.Module):
    """creates a convolutional layer with optional maxpool, batchnorm, and dropout"""
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                 batchnorm=True, maxpool=True, maxpool_size=(2, 2), dropout=None):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding)

        if maxpool:
            self.mp = nn.MaxPool2d(maxpool_size, stride=maxpool_size)
        else:
            self.mp = None

        if batchnorm:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None

        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        # self.init_weights()

    def forward(self, nn_input):

        x = nn_input

        if self.bn:
            x = F.relu(self.bn(self.conv(x)))
        else:
            x = F.relu(self.conv(x))
        if self.mp:
            x = self.mp(x)
        if self.dropout:
            x = self.dropout(x)

        return x


class AudioCNN(nn.Module):
    """Convolutions over spectrogram; merges with VGG-ish embeddings for fully-connected layers"""
    def __init__(self):
        super(AudioCNN, self).__init__()

        dropout = .5

        # spectrogram convolutions
        self.conv_block_1 = ConvBlock(in_channels=1, out_channels=8, kernel_size=(1, 1), stride=(1, 1),
                                       padding=(0, 0), batchnorm=True, maxpool=False, maxpool_size=None,
                                      dropout=dropout)

        self.conv_block_2 = ConvBlock(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=(2, 2),
                                       padding=(1, 1), batchnorm=True, maxpool=False, maxpool_size=None,
                                      dropout=dropout)

        self.conv_block_3 = ConvBlock(in_channels=16, out_channels=32, kernel_size=(7, 7), stride=(2, 2),
                                       padding=(1, 1), batchnorm=True, maxpool=True, maxpool_size=(2, 2),
                                      dropout=dropout)

        self.conv_block_4 = ConvBlock(in_channels=32, out_channels=64, kernel_size=(5, 5), stride=(2, 2),
                                       padding=(1, 1), batchnorm=True, maxpool=False, maxpool_size=None,
                                      dropout=dropout)

        self.conv_block_5 = ConvBlock(in_channels=64, out_channels=128, kernel_size=(5, 5), stride=(2, 2),
                                       padding=(1, 1), batchnorm=True, maxpool=False, maxpool_size=None,
                                      dropout=dropout)

        self.conv_block_6 = ConvBlock(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(2, 2),
                                       padding=(1, 1), batchnorm=True, maxpool=False, maxpool_size=None,
                                      dropout=dropout)

        ## combine output of conv_block_6 with VGG-ish embedding
        self.fc1 = nn.Linear(256, 128, bias=True)
        self.fc1_bn = nn.BatchNorm1d(128)
        self.fc_final = nn.Linear(128, NUM_CLASSES, bias=True)

        self.fc_dropout = nn.Dropout(.2)

    def forward(self, nn_input):

        x = nn_input  # spectrogram
        # print(x.shape)
        # spectrogram convolutions
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = self.conv_block_4(x)
        x = self.conv_block_5(x)
        x = self.conv_block_6(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        ## fully-connected layers
        x = self.fc1(x)
        x = self.fc1_bn(x)
        x = F.relu(x)
        x = self.fc_dropout(x)
        x = self.fc_final(x)
        output = F.softmax(x)

        return output


if __name__ == '__main__':

    train_dir = "train_files"
    test_dir = "test_files"

    TRAIN = AudioDataset(train_dir)
    TEST = AudioDataset(test_dir)

    TRAIN_LOADER = DataLoader(dataset=TRAIN, batch_size=BATCH_SIZE, shuffle=True)
    TEST_LOADER = DataLoader(dataset=TEST, batch_size=BATCH_SIZE, shuffle=True)

    model = AudioCNN().to(device)

    ## if training from checkpoint, ensure checkpoint matches model class architecture
    if CHECKPOINT:
        checkpoint = torch.load(CHECKPOINT_PATH)
        model.load_state_dict(checkpoint)

    ## Training params
    starting_lr = .01 #.001
    lr = starting_lr
    min_lr = 1e-6
    stagnation = 0
    stagnation_threshold = 10
    reduce_lr_rate = .1
    running_loss = 0
    best_val_loss = 10
    train_losses, test_losses = [], []

    ## Loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=starting_lr)

    ## Train model
    for epoch in range(MAX_NUM_EPOCHS):
        epoch_losses = []
        epoch_losses_val = []
        if stagnation > stagnation_threshold:
            if lr <= min_lr:
                lr = starting_lr
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                reduce_lr_rate += .1
                print('.' * 50)
                print('reset learning rate to', lr)
                print('.' * 50)
                stagnation = 0
            else:
                lr = lr * reduce_lr_rate
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                print('.' * 50)
                print('reduced learning rate to', lr)
                print('.' * 50)
                stagnation = 0
        for i, (spectrogram, label) in enumerate(TRAIN_LOADER):
            # Forward pass
            outputs = model(spectrogram)
            loss = criterion(outputs, label)
            epoch_losses.append(loss.item())
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Epoch:', epoch)
        print('lr:', lr)
        print('Train loss:', np.mean(np.array(epoch_losses)))
        test_loss = 0
        accuracy = 0
        model.eval()

        ## get validation set loss
        with torch.no_grad():
            epoch_losses_val = []
            for i, (spectrogram, label) in enumerate(TRAIN_LOADER):
                # Forward pass
                outputs = model(spectrogram)
                loss = criterion(outputs, label)
                epoch_losses_val.append(loss.item())
            val_loss = np.mean(np.array(epoch_losses_val))
            print('val loss:', val_loss)

            if val_loss < best_val_loss:
                stagnation = 0
                best_val_loss = val_loss
                torch.save(model.state_dict(),
                           f'models/{RUN_NAME}_val_loss={val_loss:.4f}.ckpt')
            else:
                stagnation += 1
                print('Stagnation:', stagnation)
            print('best_val_loss:', best_val_loss)
            print()
            model.train()