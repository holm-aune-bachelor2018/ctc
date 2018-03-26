import librosa
import numpy as np



def mfcc(filepath, frame_length, hop_length):
    y, sr = librosa.load(filepath)
    frames = librosa.util.frame(y=y, frame_length=frame_length, hop_length=hop_length)

    list = [librosa.feature.mfcc(y=frames[i], sr=sr, n_mfcc=12) for i in range(0, frames.shape[0])]

    mfcc = np.empty([12,0])

    for i in range (0, frames.shape[0]):
        mfcc = np.insert(mfcc, i, list[i].T, axis=1)

        #print 'frame: ', list[i]

    return (mfcc)
