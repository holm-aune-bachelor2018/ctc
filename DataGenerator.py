# Taken from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html
# and modified to fit data

import soundfile as sf
import numpy as np
import data
from librosa.feature import mfcc
from keras.preprocessing.sequence import pad_sequences
from keras.utils import Sequence
from random import shuffle as shuf

# import librosa.display
# import matplotlib.pyplot as plt


class DataGenerator(Sequence):
    """
    Thread safe data generator for the fit_generator

    Args:
        df (Pandas.Dataframe): dataframes containing (filename, filesize, transcript)
        batch_size (int): size of each batch
        frame_length (int): size of each frame (samples per frame)
        hop_length (int): how far to move center of each frame when splitting audio time series
        mfcc_features (int): how many mfcc-features to extract for each frame
        epoch_length (int, default=0): the number of batches in each epoch, if set to zero it uses all available data
        shuffle (boolean, default=True): whether to shuffle the indexes in each batch

    Note:
        If hop_length is shorter than frame_length it creates overlapping frames

        See https://keras.io/utils/ - Sequence for more details on using Sequence

    """
    n_mels = 40  # mel-filterbanks per frame

    def __init__(self, df, batch_size, frame_length, hop_length, mfcc_features, epoch_length=0, shuffle=True):
        self.df = df.copy()
        self.batch_size = batch_size
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.mfcc_features = mfcc_features
        self.epoch_length = epoch_length
        self.shuffle = shuffle

        # Initializing indexes
        self.indexes = np.arange(len(self.df))

    def __len__(self):
        """Denotes the number of batches per epoch"""
        if (self.epoch_length == 0) | (self.epoch_length > int(np.floor(self.df.shape[0]/self.batch_size))):
            self.epoch_length = int(np.floor(self.df.shape[0] / self.batch_size))
        return self.epoch_length

    def __getitem__(self, batch_index):
        """
        Generates a batch of correctly shaped X and Y data

        Args:
            batch_index (int): index of the batch to generate

        Returns:
            dictionary: {
                'the_input':     np.ndarray[shape=(batch_size, max_seq_length, mfcc_features)]: input audio data
                'the_labels':    np.ndarray[shape=(batch_size, max_transcript_length)]: transcription data
                'input_length':  np.ndarray[shape=(batch_size, 1)]: length of each sequence (numb of frames) in x_data
                'label_length':  np.ndarray[shape=(batch_size, 1)]: length of each sequence (numb of letters) in y_data
            }
            dictionary: {
                'ctc':           np.ndarray[shape=(batch_size, 1)]: dummy data for dummy loss function
            }
        """

        # Generate indexes of the batch
        indexes_in_batch = self.indexes[batch_index * self.batch_size:(batch_index + 1) * self.batch_size]

        # Shuffle indexes in batch
        if (self.shuffle):
            shuf(indexes_in_batch)

        # Initializing vectors
        x_data_raw = []
        y_data_unpadded = []
        len_y_seq = []
        len_x_seq = []
        sr = 0

        # loads wav-files and transcripts
        for i in indexes_in_batch:
            # Read sound data
            path = self.df.iloc[i]['filename']
            frames, sr = sf.read(path)
            x_data_raw.append(frames)

            # Read transcript data
            y_txt = self.df.iloc[i]['transcript']
            y_int = data.text_to_int_sequence(y_txt)
            y_data_unpadded.append(y_int)

            # Save length of transcripts for the CTC
            len_y_seq.append(len(y_int))

        # Finds longest frame in batch for padding
        max_x_length = self.get_seq_size(max(x_data_raw, key=len), sr)
        x_data = np.empty([0, max_x_length, self.mfcc_features])

        # Extract mfcc features and pad so every frame-sequence is equal length
        for i in range (0,len(x_data_raw)):
            x, x_len = self.mfcc(x_data_raw[i], sr, max_x_length)
            x_data = np.insert(x_data, i, x, axis=0)
            len_x_seq.append(x_len - 2)     # -2 because ctc discards the first two outputs of the rnn network

        # Finds longest sequence in y for padding
        max_y_length = len(max(y_data_unpadded, key=len))
        y_data = np.empty([0, max_y_length])

        # Pads every sequence in y to be equal length
        for i in range(0, len(y_data_unpadded)):
            y = y_data_unpadded[i]

            for j in range(len(y), max_y_length):
                y.append(0)

            y_data = np.insert(y_data, i, y, axis=0)

        # Convert to numpy array
        input_length = np.array(len_x_seq)  # batch_size * 1
        label_length = np.array(len_y_seq)  # batch_size * 1

        # print "x_data shape: ", x_data.shape
        # print "y_data shape: ", y_data.shape
        # print "input_length shape: ", input_length.shape
        # print "label_length shape: ", label_length.shape
        # print "input length: ", input_length
        # print "label_length: ", label_length

        inputs = {'the_input': x_data,
                  'the_labels': y_data,
                  'input_length': input_length,
                  'label_length': label_length}

        outputs = {'ctc': np.zeros([self.batch_size])} # dummy data for dummy loss function

        return inputs, outputs

    def mfcc(self, frames, sr, max_pad_length):
        """
        Generates MFCC (mel frequency cepstral coefficients) and zero-pads with max_pad_length

        Args:
            frames (np.ndarray[shape=(n,)]):    audio time series
            sr (int):                           sampling rate of frames
            max_pad_length (int):               length (number of frames) of longest sequence in batch

        Returns:
            np.ndarray[shape=(max_seq_length, mfcc_features)]: padded mfcc features of audio time series
            int: length of sequence before padding (input for CTC)
        """

        mfcc_frames = mfcc(frames, sr, n_fft=self.frame_length, hop_length=self.hop_length,
                           n_mfcc=self.mfcc_features, n_mels=self.n_mels)

        x_length = mfcc_frames.shape[1]
        mfcc_frames = pad_sequences(mfcc_frames, maxlen=max_pad_length, dtype='float',
                                    padding='post', truncating='post')
        mfcc_frames = mfcc_frames.T

        return mfcc_frames, x_length

    def get_seq_size(self, frames, sr):
        """
        Get audio sequence size of audio converted to mfcc-features

        Args:
            frames (np.ndarray [shape=(n,)]): audio time series
            sr (int): sampling rate of frames

        Returns:
            int: sequence size of mfcc-converted audio
        """
        mfcc_frames = mfcc(frames, sr, n_fft=self.frame_length, hop_length=self.hop_length,
                           n_mfcc=self.mfcc_features, n_mels=self.n_mels)
        return mfcc_frames.shape[1]

"""
# Plots mfcc
def plot_mfcc(mfcc_frames):
    print "\n Plotting mfcc with shape: ", mfcc_frames.shape
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc_frames, x_axis='time')
    print "librosa display... "
    plt.colorbar()
    plt.title('MFCC')
    plt.tight_layout()
    plt.interactive(False)
    plt.show()
"""