from keras.models import Model
import nn_models
import preprocessing
import numpy as np
from keras import optimizers
import pandas as pd
import data

#Preprocessing
path = "sample_data/wav_sample/sample_librivox-dev-clean.csv"
frequency = 16
frame_length = 20*frequency
hop_length = 10*frequency

dataprop, df_final = data.combine_all_wavs_and_trans_from_csvs(path)
dataX, dataY, x_length, y_length = preprocessing.generate_batch(df_final, frame_length, hop_length)

df_final.to_csv('data.csv')


# Model

units = 512                         #numb of hidden nodes
mfcc_features = 12                  #input_dim
input_shape=(None, mfcc_features)   #None to be able to process batches of any size

np.random.seed(7)
loss = {'ctc': lambda y_true, y_pred: y_pred}
optimizer = optimizers.Adam(clipvalue=0.5)
metrics = ['accuracy']

n_batch = len(dataX)
n_epoch = 1


brnn_model = nn_models.dnn_brnn(units,)
brnn_model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
brnn_model.summary()

#dataX
# x_data
# y_true - (samples, max_string_length) containing the truth labels.
# input_length - (samples, 1) containing the sequence length for each batch item in y_pred = ???
# label_length - (samples, 1) containing the sequence length for each batch item in y_true = 132

# CTC requirement: label_length must be shorter than the input_length


x_data = dataX                        # batch_size * time_steps * features
y_true = dataY                        # batch_size * max_string_length
input_length = np.array(x_length)     # batch_size * 1
label_length = np.array(y_length)     # batch_size * 1




print "\Before fitting: "
print "x_data shape: ", x_data.shape
print "y_data shape: ", y_true.shape
print "input_length shape: ", input_length.shape
print "label_length shape: ", label_length.shape, "\n"


#print "x_data: \n", x_data
#print "y_true: \n", y_true
#print "label length: \n", label_length
#print "input length: \n", input_length

#y_true, y_pred, input_length, label_length
for i in range(n_epoch):
    brnn_model.fit([x_data, y_true, input_length, label_length], dataY, epochs=n_epoch, batch_size=n_batch, verbose=2)
    brnn_model.reset_states()

#Results
#scores = nn_models.evaluate(X, y, verbose=0)
