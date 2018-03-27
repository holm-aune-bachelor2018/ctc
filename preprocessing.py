import librosa
import numpy as np
import pandas as pd

# generates a batch from DataFrames
def generate_batch(df, frame_length, hop_length):

    dataX = []
    dataY = []

    # dataX: append x to dataX and pad
    for i in range(0, df.size):
        path = df.iloc[i]['wav_filename']
        print "Path: ", path
        y_txt = df.iloc[i]['transcript']
        print "Transc: ", y_txt
        x = mfcc(path, frame_length, hop_length)

        #dataX.insert(dataX, i, x)

    print 'y: ', y_txt
    print 'x: ', x

    # dataY: y_txt to integer and then to correct shape
    return dataX, dataY


# generates MFCC (mel frequency cepstral coefficients)
def mfcc(filepath, frame_length, hop_length):
    y, sr = librosa.load(filepath)
    frames = librosa.util.frame(y=y, frame_length=frame_length, hop_length=hop_length)

    list = [librosa.feature.mfcc(y=frames[i], sr=sr, n_mfcc=12) for i in range(0, frames.shape[0])]

    mfcc = np.empty([12,0])

    for i in range (0, frames.shape[0]):
        mfcc = np.insert(mfcc, i, list[i].T, axis=1)

        #print 'frame: ', list[i]

    return mfcc
