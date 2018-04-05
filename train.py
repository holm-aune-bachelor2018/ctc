import nn_models
from keras import optimizers
import data
import pandas as pd
from DataGenerator import DataGenerator


# Preprocessing
path = "sample_data/wav_sample/sample_librivox-dev-clean.csv"
frequency = 16

# df_final.to_csv('data.csv')
dataprop, df = data.combine_all_wavs_and_trans_from_csvs(path)

# TODO: validation data?
validation_df = pd.DataFrame()
validation_df = validation_df.append(df, ignore_index=True)
print dataprop, "\n"

# Parameters
params = {'batch_size': 5,
          'frame_length': 20*frequency,
          'hop_length': 10*frequency,
          'mfcc_features': 12,
          'shuffle': False}

# Generators
training_generator = DataGenerator(df, **params)
validation_generator = DataGenerator(validation_df, **params)

# Model
units = 512                                         # numb of hidden nodes
input_shape = (None, params.get('mfcc_features'))   # "None" to be able to process batches of any size
output_dim = 29                                     # output dimension (n-1)

loss = {'ctc': lambda y_true, y_pred: y_pred}

eps = 1e-8  # epsilon 1e-8
learning_rate = 0.001
optimizer = optimizers.Adam(lr=learning_rate, epsilon=eps, clipnorm=5.0)
metrics = ['accuracy']


brnn_model = nn_models.dnn_brnn(units, params.get('mfcc_features'), output_dim)
brnn_model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
brnn_model.summary()

# Train model on dataset
brnn_model.fit_generator(generator=training_generator,
                         validation_data=validation_generator,
                         epochs=1,
                         verbose=2)
#Results
#scores = nn_models.evaluate(X, y, verbose=0)
