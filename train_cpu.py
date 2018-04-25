# !/home/anitakau/envs/tensorflow-workq/bin/python
# !/home/marith1/envs/tensorflow/bin/python

import nn_models
from keras import optimizers, models
import data
from DataGenerator import DataGenerator
import keras.backend as K
from LossCallback import LossCallback
from datetime import datetime
import argparse

# Trains model on CPU

def main(args):
    # Path to training and testing/validation data

    path = "/home/<user>/ctc/data_dir/librivox-dev-clean-100.csv"
    path_validation = "/home/<user>/ctc/data_dir/librivox-test-clean.csv"

    # Create training and validation dataframes
    print "\nReading training data:"
    _, input_dataframe = data.combine_all_wavs_and_trans_from_csvs(path)
    print "\nReading validation data: "
    _, validation_df = data.combine_all_wavs_and_trans_from_csvs(path_validation)

    # input_dataframe.to_csv('data.csv')
    # validation_df.to_csv('valid_data.csv')

    # Parameters for script
    batch_size = args.batchsize
    mfcc_features = args.mfccs
    input_epoch_length = args.in_el
    epochs = args.epochs
    units = args.units
    learning_rate = args.lr
    save_model = args.save_model
    load_model = args.load_model

    # Sampling rate of data in khz (LibriSpeech is 16khz)
    frequency = 16
    shuffle = True
    dropout = 0.2

    # Data generation parameters
    params = {'batch_size': batch_size,
              'frame_length': 20 * frequency,
              'hop_length': 10 * frequency,
              'mfcc_features': mfcc_features,
              'epoch_length': input_epoch_length,
              'shuffle': shuffle
              }

    # Data generators for training data and validation data
    training_generator = DataGenerator(input_dataframe, **params)
    validation_generator = DataGenerator(validation_df, **params)

    # Model input and output shape
    input_shape = (None, params.get('mfcc_features'))  # "None" to be able to process batches of any size
    output_dim = 29  # Output dim: features to predict + 1 for the CTC blank prediction

    # Optimization algorithm used to update network weights
    eps = 1e-8  # epsilon 1e-8
    optimizer = optimizers.Adam(lr=learning_rate, epsilon=eps, clipnorm=2.0)

    # Dummy loss-function to compile model, actual CTC loss-function defined as a lambda layer in model
    loss = {'ctc': lambda y_true, y_pred: y_pred}

    # Print training data at the beginning of training
    calc_epoch_length = training_generator.__len__()
    print "\n\nModel and training parameters: "
    print "Starting time: ", datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print " - epochs: ", epochs, "\n - batch size: ", batch_size, \
        "\n - input epoch length: ", input_epoch_length, "\n - network epoch length: ", calc_epoch_length, \
        "\n - training on ", calc_epoch_length * batch_size, " files", "\n - learning rate: ", learning_rate, \
        "\n - hidden units: ", units, "\n - mfcc features: ", mfcc_features, "\n - dropout: ", dropout, "\n"

    # Train model on dataset
    if load_model:
        model = models.load_model(load_model, custom_objects={'clipped_relu': nn_models.clipped_relu})
        print ("\nLoaded existing model at: ", load_model)

    else:
        model = nn_models.dnn_brnn(units, params.get('mfcc_features'), output_dim, dropout=dropout)
        print("\nDidn't load existing model\n")

    # Print model
    model.summary()

    model.compile(loss=loss, optimizer=optimizer)
    # Creates a test function that takes sound input and outputs predictions
    # Used to calculate WER while training the network
    input_data = model.get_layer('the_input').input
    y_pred = model.get_layer('ctc').input[0]
    test_func = K.function([input_data], [y_pred])

    # The loss callback function that calculates WER while training
    loss_cb = LossCallback(test_func, validation_generator)

    model.fit_generator(generator=training_generator,
                                 epochs=epochs,
                                 verbose=2,
                                 callbacks=[loss_cb],
                                 validation_data=validation_generator,
                                 workers=1,
                                 # max_queue_size=12,
                                 shuffle=shuffle)

    if args.save_model:
        model.save(save_model)

    K.clear_session()

    print "Ending time: ", datetime.now().strftime('%Y-%m-%d %H:%M:%S')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--batchsize', type=int, default=10, help='Number of files in one batch')
    parser.add_argument('--mfccs', type=int, default=26, help='Number of mfcc features per frame to extract')
    parser.add_argument('--in_el', type=int, default=24, help='Number of batches per epoch. 0 trains on full dataset')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--units', type=int, default=64, help='Number of hidden nodes')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--save_model', type=str, help='Path to save model. If empty, no model is saved')
    parser.add_argument('--load_model', type=str, help='Path to load existing model. If empty, no model is loaded')

    args = parser.parse_args()

    print "args: ", args
    main(args)
