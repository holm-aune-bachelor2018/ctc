"""
LICENSE

This file is part of Speech recognition with CTC in Keras.
The project is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
version.
The project is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this project.
If not, see http://www.gnu.org/licenses/.

"""

import numpy as np
from keras.preprocessing.sequence import pad_sequences
from librosa.feature import mfcc, melspectrogram
from soundfile import read

from utils.text_utils import text_to_int_sequence


def load_audio(df, indexes_in_batch):
    """
    loads the the corresponding frames (audio time series) from dataframe containing filename, filesize, transcript
    :param df: dataframe containing filename, filesize, transcript
    :param indexes_in_batch: list containing the indexes of the audio filenames in the dataframe that is to be loaded
    :return: x_data_raw: list containing loaded audio time series
             y_data_raw: list containing transcripts corresponding to loaded audio
             sr: sampling rate of frames
    """
    sr = 0
    x_data_raw = []
    y_data_raw = []

    # loads wav-files and transcripts
    for i in indexes_in_batch:

        # Read sound data
        path = df.iloc[i]['filename']
        frames, sr = read(path)
        x_data_raw.append(frames)

        # Read transcript data
        y_txt = df.iloc[i]['transcript']
        y_data_raw.append(y_txt)

    return x_data_raw, y_data_raw, sr


def extract_mfcc_and_pad(frames, sr, max_pad_length, frame_length, hop_length, mfcc_features, n_mels):
    """
    Generates MFCC (mel frequency cepstral coefficients) and zero-pads with max_pad_length
    :param frames: audio time series
    :param sr: sampling rate of audio time series
    :param max_pad_length: length (no. of frames) of longest sequence in batch
    :param frame_length: length of the frames to be extracted
    :param hop_length: length of hops (for overlap)
    :param mfcc_features: number of mfcc features to extract
    :param n_mels: number of mels
    :return: mfcc_padded: padded MFCC-sequence
             x_length: unpadded length MFCC-sequence
    """

    mfcc_frames = mfcc(frames, sr, n_fft=frame_length, hop_length=hop_length, n_mfcc=mfcc_features, n_mels=n_mels)
    x_length = mfcc_frames.shape[1]
    mfcc_padded = pad_sequences(mfcc_frames, maxlen=max_pad_length, dtype='float', padding='post',
                                truncating='post')
    mfcc_padded = mfcc_padded.T

    return mfcc_padded, x_length


def extract_mel_spectrogram_and_pad(frames, sr, max_pad_length, frame_length, hop_length, n_mels):
    """
    Generates mel spectrograms and zero-pads with max_pad_length
    :param frames: audio time series
    :param sr: sampling rate of audio time series
    :param max_pad_length: length (no. of frames) of longest sequence in batch
    :param frame_length: length of the frames to be extracted
    :param hop_length: length of the hops (for overlap)
    :param n_mels: number of mels
    :return: spectrogram_padded: padded melspectrogram-sequence
             x_length: unpadded length melspectrogram-sequence
    """
    spectrogram = melspectrogram(frames, sr, n_fft=frame_length, hop_length=hop_length, n_mels=n_mels)
    x_length = spectrogram.shape[1]
    spectrogram_padded = pad_sequences(spectrogram, maxlen=max_pad_length, dtype='float', padding='post',
                                       truncating='post')
    spectrogram_padded = spectrogram_padded.T

    return spectrogram_padded, x_length


def convert_and_pad_transcripts(y_data_raw):
    """
    Converts and pads transcripts from text to int sequences
    :param y_data_raw: transcripts
    :return: y_data: numpy array with transcripts converted to a sequence of ints and zero-padded
             label_length: numpy array with length of each sequence before padding
    """
    # Finds longest sequence in y for padding
    max_y_length = len(max(y_data_raw, key=len))

    y_data = np.empty([0, max_y_length])
    len_y_seq = []

    # Converts to int and pads to be equal max_y_length
    for i in range(0, len(y_data_raw)):
        y_int = text_to_int_sequence(y_data_raw[i])
        len_y_seq.append(len(y_int))

        for j in range(len(y_int), max_y_length):
            y_int.append(0)

        y_data = np.insert(y_data, i, y_int, axis=0)

    # Convert transcript length list to numpy array
    label_length = np.array(len_y_seq)

    return y_data, label_length
