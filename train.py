#!/home/anitakau/envs/tensorflow-workq/bin/python
#!/home/marith1/envs/tensorflow/bin/python

import sys
import nn_models
from keras import optimizers
from keras.utils import multi_gpu_model
import data
from DataGenerator import DataGenerator
import keras.backend as K
from LossCallback import LossCallback
import tensorflow as tf


# Preprocessing
path = "/home/<user>/ctc/data_dir/librivox-dev-clean.csv"
path_validation = "/home/<user>/ctc/data_dir/librivox-test-clean.csv"

print "Reading training data:"
data_properties, input_dataframe = data.combine_all_wavs_and_trans_from_csvs(path)
print "\n Reading validation data: "
_, validation_df = data.combine_all_wavs_and_trans_from_csvs(path_validation)

# input_dataframe.to_csv('data.csv')
# validation_df.to_csv('validdata.csv')

# sampling rate of data in khz (LibriSpeech is 16khz)
frequency = 16

# Parameters for script
# batch_size, mfcc_features, epoch_length, epochs, units, learning_rate, model_name

batch_size = int(sys.argv[1])                       # Number of files in one batch
mfcc_features = int(sys.argv[2])                    # Number of mfcc features (per frame) to extract
input_epoch_length = int(sys.argv[3])               # Number of batches per epoch (if 0, trains on full dataset)

# Model specifications
epochs = int(sys.argv[4])                           # Number of epochs
units = int(sys.argv[5])                            # Number of hidden nodes
learning_rate = int(sys.argv[6])                    # Learning rate
model_name = sys.argv[7]                            # path to save model

params = {'batch_size': batch_size,
          'frame_length': 20 * frequency,
          'hop_length': 10 * frequency,
          'mfcc_features': mfcc_features,
          'epoch_length': input_epoch_length
}
'''
params = {'batch_size': 12,        
          'frame_length': 20 * frequency,
          'hop_length': 10 * frequency,
          'mfcc_features': 13,
          'epoch_length': 10        # number of batches per epoch (for testing)
}
'''
# Generators
training_generator = DataGenerator(input_dataframe, **params)
validation_generator = DataGenerator(validation_df, **params)

input_shape = (None, params.get('mfcc_features'))   # "None" to be able to process batches of any size
output_dim = 29                                     # output dimension (n-1)

eps = 1e-8                                          # epsilon 1e-8
learning_rate = 0.001
optimizer = optimizers.Adam(lr=learning_rate, epsilon=eps, clipnorm=2.0)

calc_epoch_length = training_generator.__len__()

print "\n\n Model and training parameters: "
print " - epochs: ", epochs, "\n - batch size: ", batch_size, \
      "\n - input epoch length: ", input_epoch_length, "\n - network epoch length: ", calc_epoch_length, \
      "\n - training on ",calc_epoch_length*batch_size," files","\n - learning rate: ", learning_rate,\
      "\n - hidden units: ", units, "\n - mfcc features: ", mfcc_features, "\n"

# loss function to compile model, actual CTC loss function defined as a lambda layer in model
loss = {'ctc': lambda y_true, y_pred: y_pred}

with tf.device('/cpu:0'):
    model = nn_models.dnn_brnn(units, params.get('mfcc_features'), output_dim)

parallel_model = multi_gpu_model(model, gpus=2)
parallel_model.compile(loss=loss, optimizer=optimizer)

model.summary()

y_pred = model.get_layer('ctc').input[0]
input_data = model.get_layer('the_input').input

test_func = K.function([input_data], [y_pred])

loss_cb = LossCallback(test_func, validation_generator)

# Train model on dataset
parallel_model.fit_generator(generator=training_generator,
                    epochs=epochs,
                    verbose=2,
                    callbacks=[loss_cb],
                    validation_data=validation_generator,
                    workers=1)

model.save(model_name)

K.clear_session()