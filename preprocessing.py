import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

sample_rate = 16000

# generates a batch from DataFrames
def generate_batch(df, frame_length, hop_length):
    dataX = np.empty([12, 0])
    dataY = []

    # dataX: append x to dataX and pad

    for i in range(0, 3):
    #for i in range(0, df.shape[0]):
        path = df.iloc[i]['wav_filename']
        y_txt = df.iloc[i]['transcript']

        x = mfcc(path, frame_length, hop_length)

        dataX = np.append(dataX, x, axis=1)

    print "Shape DataX: ", dataX.shape
    plot_mfcc(dataX)

    # dataY: y_txt to integer and then to correct shape
    return dataX, dataY


#todo: Normalize input

# generates MFCC (mel frequency cepstral coefficients)
def mfcc(file_path, frame_length, hop_length):
    y, sr = librosa.load(file_path, sr=sample_rate)

    mfcc_frames = librosa.feature.mfcc(y, sample_rate, n_fft=frame_length, hop_length=hop_length, n_mfcc=12)

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