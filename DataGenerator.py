# Taken from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html
# and modified to fit data

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import data
from keras.preprocessing.sequence import pad_sequences
import keras

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, df, batch_size, frame_length, hop_length, mfcc_features, shuffle=True):
        'Initialization'
        self.df = df
        self.batch_size = batch_size
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.mfcc_features = mfcc_features
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        epoch_length = int(np.floor(self.df.shape[0] / self.batch_size))
        return epoch_length

    def __getitem__(self, index):
        """
        Generates a batch of correctly shaped X and Y data from given dataframe
        :param df: Dataframes containing (wav_filename, wav_filesize, transcript)
        :param frame_length: Length of each frame generated.
        :param hop_length: How far to jump for each frame.
        :param mfcc_features:
        :return:    x_data:     dim(batch_size * max_timesteps * mfcc_features)
                    y_data:     dim(batch_size * max_transcript_length)
                    len_x_seq:  dim(batch_size * 1) length of each sequence (numb of frames) in x_data
                    len_y_seq:  dim(batch_size * 1) length of each sequence (numb of letters) in y_data
        """
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Initializing vectors
        x_data_raw = []
        y_data_unpadded = []

        len_y_seq = []
        sr = 0

        # loads wav-files and transcripts
        for i in indexes:
            path = self.df.iloc[i]['wav_filename']
            frames, sr = librosa.load(path, sr=None)
            x_data_raw.append(frames)

            y_txt = self.df.iloc[i]['transcript']
            y_int = data.text_to_int_sequence(y_txt)
            y_data_unpadded.append(y_int)

            len_y_seq.append(len(y_int))

        # Finds longest frame in batch for padding
        max_x_length = self.get_seq_size(max(x_data_raw, key=len), sr)
        x_data = np.empty([0, max_x_length, self.mfcc_features])
        len_x_seq = []

        # Extract mfcc features and pad so every frame-sequence is equal length
        for i in range (0,len(x_data_raw)):
            x, x_len = self.mfcc(x_data_raw[i], sr, max_x_length)
            x_data = np.insert(x_data, i, x, axis=0)
            len_x_seq.append(x_len)

        # Finds longest sequence in y for padding
        max_y_length = len(max(y_data_unpadded, key=len))
        y_data = np.empty([0, max_y_length])

        # Pads every sequence in y to be equal length
        for i in range(0, len(y_data_unpadded)):
            y = y_data_unpadded[i]

            for j in range(len(y), max_y_length):
                y.append(0)

            y_data = np.insert(y_data, i, y, axis=0)

        # print "DataX shape: ", x_data.shape
        # plot_mfcc(x_data.T)
        # print "y_data: \n", y_data

        input_length = np.array(len_x_seq)  # batch_size * 1
        label_length = np.array(len_y_seq)  # batch_size * 1

        print "\nIndexes: ", indexes
        print "x_data shape: ", x_data.shape
        print "y_data shape: ", y_data.shape
        # print "input_length shape: ", input_length.shape
        # print "label_length shape: ", label_length.shape
        print "input length: ", input_length
        print "label_length: ", label_length

        inputs = {'the_input': x_data,
                  'the_labels': y_data,
                  'input_length': input_length,
                  'label_length': label_length}

        outputs = {'ctc': np.zeros([self.batch_size])} # dummy data for dummy loss function

        return inputs, outputs

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.df))
        # if self.shuffle == True:
        #    np.random.shuffle(self.indexes)

    def mfcc(self, frames, sr, max_pad_length):
        # TODO: Normalize input (between 0 and 1? -1 and 1?)
        """
        Generates MFCC (mel frequency cepstral coefficients) and zero-pads with max_pad_length
        :param frames: path to wav-file
        :param sr: wav frequency
        :param max_pad_length: length (number of frames) of longest sequence in batch
        :return: mfcc_frames: Padded 2d array with time_steps * input_dim (mfcc features)
                 x_length: length of sequence before padding (input for CTC)
        """

        mfcc_frames = librosa.feature.mfcc(frames, sr, n_fft=self.frame_length, hop_length=self.hop_length,
                                           n_mfcc=self.mfcc_features)

        x_length = mfcc_frames.shape[1]
        mfcc_frames = pad_sequences(mfcc_frames, maxlen=max_pad_length, dtype='float',
                                    padding='post', truncating='post')
        mfcc_frames = mfcc_frames.T

        return mfcc_frames, x_length

    def get_seq_size(self, frame, sr):
        mfcc = librosa.feature.mfcc(frame, sr, n_fft=self.frame_length, hop_length=self.hop_length,
                                    n_mfcc=self.mfcc_features)
        return mfcc.shape[1]


# Plots mfcc
def plot_mfcc(self, mfcc_frames):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc_frames, x_axis='time')
    plt.colorbar()
    plt.title('MFCC')
    plt.tight_layout()
    plt.interactive(False)
    plt.show()