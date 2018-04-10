import nn_models
from keras import optimizers
import data
import pandas as pd
from DataGenerator import DataGenerator
import keras.backend as K
from LossCallback import LossCallback
# Preprocessing
path = "sample_data/wav_sample/sample_librivox-dev-clean.csv"


# df_final.to_csv('data.csv')
dataprop, input_dataframe = data.combine_all_wavs_and_trans_from_csvs(path)
frequency = 16

# TODO: validation data?
validation_df = pd.DataFrame()
validation_df = validation_df.append(input_dataframe, ignore_index=True)

# Parameters
params = {'batch_size': 5,
          'frame_length': 20 * frequency,
          'hop_length': 10 * frequency,
          'mfcc_features': 26,
          'shuffle': False}

# Generators
training_generator = DataGenerator(input_dataframe, **params)
validation_generator = DataGenerator(validation_df, **params)

# Model specifications
units = 512                                         # numb of hidden nodes
input_shape = (None, params.get('mfcc_features'))   # "None" to be able to process batches of any size
output_dim = 29                                     # output dimension (n-1)

epochs = 2                                          # number of epochs

# loss function to compile model, actual CTC loss function defined as a lambda layer in model
loss = {'ctc': lambda y_true, y_pred: y_pred}

eps = 1e-8                                          # epsilon 1e-8
learning_rate = 0.001
optimizer = optimizers.Adam(lr=learning_rate, epsilon=eps, clipnorm=5.0)
metrics = ['accuracy']


model = nn_models.dnn_brnn(units, params.get('mfcc_features'), output_dim)
model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
# model.summary()

y_pred = model.get_layer('ctc').input[0]
input_data = model.get_layer('the_input').input

test_func = K.function([input_data], [y_pred])

# viz_cb = VizCallback(run_name, test_func, img_gen.next_val())
loss_cb = LossCallback(test_func, validation_generator)

# Train model on dataset
model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    epochs=epochs,
                    shuffle=True,
                    callbacks=[loss_cb],
                    verbose=2)


K.clear_session()