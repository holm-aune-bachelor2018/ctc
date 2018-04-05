import nn_models
import preprocessing
import numpy as np
from keras import optimizers
import data
from keras import backend as K

# TODO: sjekk om server viker, fiks data_generator og avtal møte med Ole


# Preprocessing
path = "sample_data/wav_sample/sample_librivox-dev-clean.csv"
frequency = 16
frame_length = 20 * frequency
hop_length = 10 * frequency         #
mfcc_features = 12                  # input_dim

index = 0
batch_size = 10

# df_final.to_csv('data.csv')
dataprop, df_final = data.combine_all_wavs_and_trans_from_csvs(path)

# Model
units = 512  # numb of hidden nodes
input_shape = (None, mfcc_features)  # "None" to be able to process batches of any size
output_dim = 29  # output dimension (n-1)

np.random.seed(7)
loss = {'ctc': lambda y_true, y_pred: y_pred}

eps = 1e-8  # epsilon 1e-8
learning_rate = 0.001
optimizer = optimizers.Adam(lr=learning_rate, epsilon=eps, clipnorm=5.0)
metrics = ['accuracy']


brnn_model = nn_models.dnn_brnn(units, mfcc_features, output_dim)
brnn_model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
brnn_model.summary()


def model_train():
    dataX, dataY, x_length, y_length = preprocessing.generate_batch(df_final, frame_length, hop_length, index, batch_size, mfcc_features)

    n_batch = batch_size
    n_epoch = 1
    #dataX
    # x_data
    # y_true - (samples, max_string_length) containing the truth labels.
    # input_length - (samples, 1) containing the sequence length for each batch item in y_pred
    # label_length - (samples, 1) containing the sequence length for each batch item in y_true

    # CTC requirement: label_length must be shorter than the input_length

    x_data = dataX                        # batch_size * time_steps * features
    y_true = dataY                        # batch_size * max_string_length
    input_length = np.array(x_length)     # batch_size * 1
    label_length = np.array(y_length)     # batch_size * 1

    print "\nBefore fitting: "
    print "x_data shape: ", x_data.shape
    print "y_data shape: ", y_true.shape
    print "input_length shape: ", input_length.shape
    print "label_length shape: ", label_length.shape, "\n"


    for i in range(n_epoch):
        brnn_model.fit([x_data, y_true, input_length, label_length], dataY, epochs=n_epoch, batch_size=n_batch, verbose=2)
        brnn_model.reset_states()

#Results
#scores = nn_models.evaluate(X, y, verbose=0)
