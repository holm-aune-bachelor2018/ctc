import numpy as np
from keras.preprocessing.sequence import pad_sequences
from librosa.feature import mfcc, melspectrogram
from soundfile import read

from utils.text_utils import text_to_int_sequence


def load_audio(df, indexes_in_batch):
    """

    :param df:
    :param indexes_in_batch:
    :return:
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

    Args:
        frames (np.ndarray[shape=(n,)]):    audio time series
        sr (int):                           sampling rate of frames
        max_pad_length (int):               length (number of frames) of longest sequence in batch
        frame_length
        hop_length
        mfcc_features
        n_mels

    Returns:
        np.ndarray[shape=(max_seq_length, mfcc_features)]: padded mfcc features of audio time series
        int: length of sequence before padding (input for CTC)
    """

    mfcc_frames = mfcc(frames, sr, n_fft=frame_length, hop_length=hop_length, n_mfcc=mfcc_features, n_mels=n_mels)
    x_length = mfcc_frames.shape[1]
    mfcc_padded = pad_sequences(mfcc_frames, maxlen=max_pad_length, dtype='float', padding='post',
                                truncating='post')
    mfcc_padded = mfcc_padded.T

    return mfcc_padded, x_length


def extract_mel_spectrogram_and_pad(frames, sr, max_pad_length, frame_length, hop_length, n_mels):
    """

    :param frames:
    :param sr:
    :param max_pad_length:
    :param frame_length:
    :param hop_length:
    :param n_mels:
    :return:
    """
    spectrogram = melspectrogram(frames, sr, n_fft=frame_length, hop_length=hop_length, n_mels=n_mels)
    x_length = spectrogram.shape[1]
    spectrogram_padded = pad_sequences(spectrogram, maxlen=max_pad_length, dtype='float', padding='post',
                                       truncating='post')
    spectrogram_padded = spectrogram_padded.T

    return spectrogram_padded, x_length


def convert_and_pad_transcripts(y_data_raw):
    """

    :param y_data_raw:
    :return:
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
