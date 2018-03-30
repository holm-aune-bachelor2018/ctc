from keras.models import Model
import nn_models
import preprocessing
import numpy as np
from keras import Input
import pandas as pd
import data

#Preprocessing
path = "sample_data/wav_sample/sample_librivox-dev-clean.csv"
frequency = 16
frame_length = 20*frequency
hop_length = 10*frequency

dataprop, df_final = data.combine_all_wavs_and_trans_from_csvs(path)
dataX, dataY = preprocessing.generate_batch(df_final, frame_length, hop_length)

print 'dataY: \n', dataY.shape
#df_final.to_csv('data.csv')


# Model

units = 512                         #numb of hidden nodes
mfcc_features = 12                  #input_dim
input_shape=(None, mfcc_features)   #None to be able to process batches of any size

np.random.seed(7)
#loss = 'categorical_crossentropy'
optimizer = 'adam'
metrics = ['accuracy']

n_batch = len(dataX)
n_epoch = 1


# lstm.simpleLSTM(units, input_shape=input_shape, X=dataX, Y=dataY)


brnn_model = nn_models.dnn_brnn(units)
brnn_model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizer, metrics=metrics)
brnn_model.summary()

#dataX
# x_data
# y_true - (samples, max_string_length) containing the truth labels.
# input_length - (samples, 1) containing the sequence length for each batch item in y_pred = ???
# label_length - (samples, 1) containing the sequence length for each batch item in y_true = 132

# numpy array? with correct shapes
# input_to_model = [tensor(), tensor(), tensor(), tensor()]

# input_to_model = [Input(name='x_data', tensor=(dataX)),
#    Input(name='y_true', shape=[None]),
#    Input(name='y_pred_len', shape=[1]),
#    Input(name='y_true_len', shape=[1])]

for i in range(n_epoch):
    brnn_model.fit(dataX, dataY, epochs=n_epoch, batch_size=n_batch, verbose=0)
    brnn_model.reset_states()



#model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

#models.fit(X, y, epochs=100, batch_size=1, verbose=0)



#Results
#scores = nn_models.evaluate(X, y, verbose=0)
