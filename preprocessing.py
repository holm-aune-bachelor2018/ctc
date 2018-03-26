import librosa
import numpy as np
import pandas as pd


def generate_batch(df, frame_length, hop_length):

    dataX = []
    dataY = []

    #for i in range(0, df):
    path = df.iloc[0]['wav_filename']
    y_txt = df.iloc[0]['transcript']
    x = mfcc(path, frame_length, hop_length)
    print 'y: ', y_txt
    print 'x: ', x
    # dataX: append x to dataX and pad
    # dataY: y_txt to integer and then to correct shape
    return dataX, dataY



def mfcc(filepath, frame_length, hop_length):
    y, sr = librosa.load(filepath)
    frames = librosa.util.frame(y=y, frame_length=frame_length, hop_length=hop_length)

    list = [librosa.feature.mfcc(y=frames[i], sr=sr, n_mfcc=12) for i in range(0, frames.shape[0])]

    mfcc = np.empty([12,0])

    for i in range (0, frames.shape[0]):
        mfcc = np.insert(mfcc, i, list[i].T, axis=1)

        #print 'frame: ', list[i]

    return (mfcc)
