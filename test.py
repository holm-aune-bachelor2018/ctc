import preprocessing
import lstm
import numpy as np


#Preprocessing
path = "data/LibriSpeech/dev-clean-wav/84-121123-0000.wav"
frequency = 16
frame_length = 20*frequency
hop_length = 10*frequency

# TODO:
# how to preprocess y data?
# final shape of X? batch handling?
y_txt = "GO DO YOU HEAR" #temporary y data
dataX = preprocessing.mfcc(path, frame_length, hop_length)
dataY = y_txt


#Model
units = 512                         #numb of hidden nodes
mfcc_features = 12                  #input_dim
input_shape=(None, mfcc_features)   #None to be able to process batches of any size

lstm.simpleLSTM(units, input_shape=input_shape, X=dataX, Y=dataY)

