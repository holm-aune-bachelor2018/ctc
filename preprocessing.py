import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import data
from keras.preprocessing.sequence import pad_sequences


def generate_batch(df, frame_length, hop_length, mfcc_features=12):
    """
    Generates a batch of correctly shaped X and Y data from given dataframe
    :param df: Dataframes contaning (wav_filename, wav_filesize, transcript)
    :param frame_length: Length of each frame generated.
    :param hop_length: How far to jump for each frame.
    :return: dataX: (batch_size * max_timesteps * mfcc_features) dim matrix,
             dataY: (batch_size * max_transcript_length) dim matrix
    """
    print "\nIs null: \n", df.isnull().any(), "\n"

    # Fetch lagrest wav_file to find the longest sequence length in batch
    largest_index = df['wav_filesize'].idxmax()
    #print 'Largest ind: ', largest_index
    path_to_largest = df.loc[largest_index]['wav_filename']
    #print "path to largest: ", path_to_largest
    max_length = get_seq_size(path_to_largest, frame_length, hop_length)
    #print "Length (in frames) of largest wav_file: ", max_length

    dataX = np.empty([0, max_length, mfcc_features])
    x_lengths = []
    listY = []
    y_lengths = []


    for i in range(0, df.shape[0]):
        # dataY: y_txt from string to integer
        y_txt = df.iloc[i]['transcript']
        y_int = data.text_to_int_sequence(y_txt)
        y_lengths.append(len(y_int))
        listY.append(y_int)

        # dataX: extract mfcc features and pad so every frame-sequence is equal length
        path = df.iloc[i]['wav_filename']
        x, x_len = mfcc(path, frame_length, hop_length, max_length, mfcc_features=mfcc_features)
        x_lengths.append(x_len)
        dataX = np.insert(dataX, i, x, axis=0)

    y_length = len(max(listY, key=len))
    # print 'Y-length: ',y_length


    dataY = np.empty([0, y_length])
    for i in range(0, df.shape[0]):
        y = listY[i]
        for j in range(len(y), y_length):
            y.append(0)

        dataY = np.insert(dataY, i, y, axis=0)


    # print "DataX shape: ", dataX.shape
    # plot_mfcc(dataX.T)
    # print "dataY: \n", dataY

    return dataX, dataY, x_lengths, y_lengths


def mfcc(file_path, frame_length, hop_length, max_pad_length, mfcc_features=12):
    # TODO: Normalize input (between 0 and 1? -1 and 1?)
    """
    Generates MFCC (mel frequency cepstral coefficients)
    :param file_path: path to wav-file
    :param frame_length: length of each frame generated
    :param hop_length: how far to jump for each frame
    :param max_pad_length: length (number of frames) of longest sequence
    :return: Padded 2d array with time_steps * input_dim (mfcc features)
    """
    y, sr = librosa.load(file_path, sr=None)

    mfcc_frames = librosa.feature.mfcc(y, sr, n_fft=frame_length, hop_length=hop_length, n_mfcc=mfcc_features)
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


def get_seq_size(path, frame_length, hop_length):
    """
    Calculates the number of frames (time-steps) for a given wav_file.
    :param path: path to wav-file
    :param frame_length: length of each frame generated
    :param hop_length: how far to jump for each frame
    :return: number of frames/sequence length
    """

    y, sr = librosa.load(path, sr=None)
    mfcc = librosa.feature.mfcc(y, sr, n_fft=frame_length, hop_length=hop_length, n_mfcc=12)

    return mfcc.shape[1]