import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import data
from keras.preprocessing.sequence import pad_sequences

sample_rate = 16000

# generates a batch from DataFrames
def generate_batch(df, frame_length, hop_length):
    dataX = np.empty([12, 0])
    dataY = []

    # dataX: append x to dataX and pad

    largest_index = df['wav_filesize'].idxmax()
    print "Size: ", largest_index

    path_to_largest = df.iloc[largest_index]['wav_filename']
    largest_file = librosa.load(path_to_largest,sample_rate)
    #duration = librosa.core.get_duration(largest_file, sr=sample_rate)
    #print "Duration: ", duration
    print "LArgest ", largest_file.shape
    for i in range(0, 3):
    #for i in range(0, df.shape[0]):


        # indeks reset
        path = df.iloc[i]['wav_filename']
        y_txt = df.iloc[i]['transcript']

        y_int = data.text_to_int_sequence(y_txt)
        x = mfcc(path, frame_length, hop_length, )

        dataX = np.append(dataX, x, axis=1)
        dataY = np.append(dataY, y_int, axis=0)


    plot_mfcc(dataX)


    # dataY: y_txt to integer and then to correct shape
    return dataX, dataY


#todo: Normalize input

def mfcc(file_path, frame_length, hop_length, max_pad_length):
    '''
    Generates MFCC (mel frequency cepstral coefficients)
    :param file_path:
    :param frame_length:
    :param hop_length:
    :param max_pad_length:
    :return: Padded 2d array with time_steps * input_dim (mfcc features)
    '''
    y, sr = librosa.load(file_path, sr=sample_rate)

    mfcc_frames = librosa.feature.mfcc(y, sample_rate, n_fft=frame_length, hop_length=hop_length, n_mfcc=12)

    mfcc_frames = mfcc_frames.T
    mfcc_frames = pad_sequences(mfcc_frames, maxlen=max_pad_length, dtype='float', padding='post', truncating='post')

    return mfcc_frames


# Plots mfcc
def plot_mfcc(mfcc_frames):

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc_frames, x_axis='time')
    plt.colorbar()
    plt.title('MFCC')
    plt.tight_layout()
    plt.interactive(False)
    plt.show()