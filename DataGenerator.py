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

    #     def generate_batch(df, frame_length, hop_length, index, batch_size, mfcc_features=12):
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        #list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        #X, y = self.__data_generation(list_IDs_temp)

        """
        Generates a batch of correctly shaped X and Y data from given dataframe
        :param df: Dataframes contaning (wav_filename, wav_filesize, transcript)
        :param frame_length: Length of each frame generated.
        :param hop_length: How far to jump for each frame.
        :param mfcc_features:
        :return:    x_data:     dim(batch_size * max_timesteps * mfcc_features)
                    y_data:     dim(batch_size * max_transcript_length)
                    len_x_seq:  dim(batch_size * 1) length of each sequence (numb of frames) in x_data
                    len_y_seq:  dim(batch_size * 1) length of each sequence (numb of letters) in y_data
        """
        # Fetch largest wav_file to find the longest sequence length in batch
        largest_index = self.df['wav_filesize'].idxmax()
        path_to_largest = self.df.loc[largest_index]['wav_filename']
        max_length = self.get_seq_size(path_to_largest)

        # print 'Largest ind: ', largest_index
        # print "path to largest: ", path_to_largest
        # print "Length (in frames) of largest wav_file: ", max_length

        # print "df inni: ", self.df

        # Initializing vectors
        x_data = np.empty([0, max_length, self.mfcc_features])
        y_data_unpadded = []
        len_x_seq = []
        len_y_seq = []
        count = 0

        for i in indexes:
            # y_data: y_txt from string to integer
            y_txt = self.df.iloc[i]['transcript']
            y_int = data.text_to_int_sequence(y_txt)
            y_data_unpadded.append(y_int)

            len_y_seq.append(len(y_int))

            # x_data: extract mfcc features and pad so every frame-sequence is equal length
            path = self.df.iloc[i]['wav_filename']
            x, x_len = self.mfcc(path, max_length)
            x_data = np.insert(x_data, count, x, axis=0)
            count += 1

            len_x_seq.append(x_len)

        # Finds longest sequence in y
        y_length = len(max(y_data_unpadded, key=len))
        y_data = np.empty([0, y_length])
        # print 'Y-length: ', y_length

        # Pads every sequence in y to be equal length
        for i in range(0, len(y_data_unpadded)):
            y = y_data_unpadded[i]

            for j in range(len(y), y_length):
                y.append(0)

            y_data = np.insert(y_data, i, y, axis=0)

        # print "DataX shape: ", x_data.shape
        # plot_mfcc(x_data.T)
        # print "y_data: \n", y_data

        input_length = np.array(len_x_seq)  # batch_size * 1
        label_length = np.array(len_y_seq)  # batch_size * 1

        # print "\nBefore fitting: "
        # print "x_data shape: ", x_data.shape
        # print "y_data shape: ", y_data.shape
        # print "input_length shape: ", input_length.shape
        # print "label_length shape: ", label_length.shape, "\n"
        # print "input length: ", input_length
        # print "label_length: ", label_length

        # Input to model with ctc: [x_data, y_true, input_length, label_length]

        inputs = {'x_data': x_data,
                  'y_true': y_data,
                  'input_length': input_length,
                  'label_length': label_length}

        outputs = {'y_data': y_data}

        return inputs, y_data

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.df))
        # if self.shuffle == True:
        #    np.random.shuffle(self.indexes)

    def mfcc(self, file_path, max_pad_length):
        # TODO: Normalize input (between 0 and 1? -1 and 1?)
        """
        Generates MFCC (mel frequency cepstral coefficients)
        :param file_path: path to wav-file
        :param max_pad_length: length (number of frames) of longest sequence
        :return: mfcc_frames: Padded 2d array with time_steps * input_dim (mfcc features)
                 x_length: length of sequence before padding
        """
        y, sr = librosa.load(file_path, sr=None)

        mfcc_frames = librosa.feature.mfcc(y, sr, n_fft=self.frame_length, hop_length=self.hop_length, n_mfcc=self.mfcc_features)
        x_length = mfcc_frames.shape[1]
        mfcc_frames = pad_sequences(mfcc_frames, maxlen=max_pad_length, dtype='float', padding='post', truncating='post')
        mfcc_frames = mfcc_frames.T

        return mfcc_frames, x_length

    # Plots mfcc
    def plot_mfcc(mfcc_frames):

        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mfcc_frames, x_axis='time')
        plt.colorbar()
        plt.title('MFCC')
        plt.tight_layout()
        plt.interactive(False)
        plt.show()

    def get_seq_size(self, path):
        """
        Calculates the number of frames (time-steps) for a given wav_file.
        :param path: path to wav-file
        :return: number of frames/sequence length
        """

        y, sr = librosa.load(path, sr=None)
        mfcc = librosa.feature.mfcc(y, sr, n_fft=self.frame_length, hop_length=self.hop_length, n_mfcc=self.mfcc_features)

        return mfcc.shape[1]