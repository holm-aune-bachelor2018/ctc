import preprocessing

import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import np_utils

numpy.random.seed(7)

# Make script later
path = "sample_data/dev-clean/84/121123/84-121123-0000.flac"
frame_length = 400
hop_length = 160

dataX = []
dataY = []

X = numpy.array()
y = numpy.array()

y_txt = "GO DO YOU HEAR" #temporary y data

mfcc_data = preprocessing.mfcc(path, frame_length, hop_length)

# Do proper input formatting/reshaping later


# Model compiling and fit
model = Sequential()

model.add(LSTM(512, input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(y.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X,y, epochs=100, batch_size=1, verbose=0)

#Results
scores = model.evaluate(X, y, verbose=0)

