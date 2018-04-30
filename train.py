# !/home/anitakau/envs/tensorflow-workq/bin/python
# !/home/marith1/envs/tensorflow/bin/python

import nn_models
from keras import models
from keras.optimizers import Adam
from keras.utils import multi_gpu_model
from data import combine_all_wavs_and_trans_from_csvs
from DataGenerator import DataGenerator
import keras.backend as K
from LossCallback import LossCallback
import tensorflow as tf
from datetime import datetime
import argparse

def main(args):
    # Path to training and testing/validation data
    #path = "/home/<user>/ctc/data_dir/librivox-train-clean-100.csv"
    #path_validation = "/home/<user>/ctc/data_dir/librivox-test-clean.csv"
    path = "data_dir/librivox-dev-clean.csv"
    path_validation = "data_dir/librivox-test-clean.csv"

    # Create training and validation dataframes
    print "\nReading training data:"
    _, input_dataframe = combine_all_wavs_and_trans_from_csvs(path)
    print "\nReading validation data: "
    _, validation_df = combine_all_wavs_and_trans_from_csvs(path_validation)

    # Input parameters for script
    # Model params:
    batch_size = args.batchsize
    mfcc_features = args.mfccs
    input_epoch_length = args.in_el
    epochs = args.epochs
    units = args.units
    learning_rate = args.lr
    dropout = args.dropout

    # Training params:
    model_type = args.model_type
    model_save = args.model_save
    model_load = args.model_load
    load_multi = args.load_multi
    checkpoint = args.checkpoint
    num_gpu = args.num_gpu
    shuffle = args.shuffle

    # Sampling rate of data in khz (LibriSpeech is 16khz)
    frequency = 16

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

    # Model output shape
    output_dim = 29  # Output dim: features to predict + 1 for the CTC blank prediction

    # Optimization algorithm used to update network weights
    eps = 1e-8  # epsilon 1e-8
    optimizer = Adam(lr=learning_rate, epsilon=eps, clipnorm=2.0)

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

    # Load previous model or create new. With device cpu ensures that the model is created/loaded on the cpu
    if model_load:
        with tf.device('/cpu:0'):
            # When loading custom objects, Keras needs to know where to find them.
            # The CTC lambda is a dummy function
            custom_objects = {'clipped_relu': nn_models.clipped_relu,
                              '<lambda>': lambda y_true, y_pred: y_pred}
            if load_multi:
                model = models.load_model(model_load, custom_objects=custom_objects)
                model = model.layers[-2]
                print ("Loaded existing model at: ", model_load)

            else:
                model = models.load_model(model_load, custom_objects=custom_objects)
                print ("Loaded existing model at: ", model_load)

    else:
        with tf.device('/cpu:0'):
            model = nn_models.model(model_type=model_type, units=units, input_dim=mfcc_features,
                                    output_dim=output_dim, dropout=dropout)
            print("Creating new model: ", model_type)

    # Train with parallel model on 2 or more gpus, must be even number
    if num_gpu > 1:
        try:
            if num_gpu % 2 == 0:
                parallel_model = multi_gpu_model(model, gpus=num_gpu)
                parallel_model.compile(loss=loss, optimizer=optimizer)

                # Print model
                model.summary()

                # Creates a test function that takes sound input and outputs predictions
                # Used to calculate WER while training the network
                input_data = model.get_layer('the_input').input
                y_pred = model.get_layer('ctc').input[0]
                test_func = K.function([input_data], [y_pred])

                # The loss callback function that calculates WER while training
                loss_cb = LossCallback(test_func, validation_generator, model, checkpoint=checkpoint,
                                       path_to_save=model_save)

                parallel_model.fit_generator(generator=training_generator,
                                             epochs=epochs,
                                             verbose=2,
                                             callbacks=[loss_cb],
                                             validation_data=validation_generator,
                                             workers=1,
                                             shuffle=shuffle)
        except ValueError:
            raise ValueError('Number of gpus must be en even number')

    elif num_gpu == 1 or num_gpu == 0:
        model.compile(loss=loss, optimizer=optimizer)
        model.summary()

        input_data = model.get_layer('the_input').input
        y_pred = model.get_layer('ctc').input[0]
        test_func = K.function([input_data], [y_pred])

        loss_cb = LossCallback(test_func, validation_generator, model, checkpoint=checkpoint,
                               path_to_save=model_save)

        model.fit_generator(generator=training_generator,
                            epochs=epochs,
                            verbose=2,
                            callbacks=[loss_cb],
                            validation_data=validation_generator,
                            workers=1,
                            shuffle=shuffle)

    else:
        raise ValueError('Error in number of gpus')

    if args.model_save:
        model.save(model_save)

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
    parser.add_argument('--model_type', type=str, default='dnn_brnn', help='What model to train: dnn_brnn, dnn_blstm')
    parser.add_argument('--model_save', type=str, help='Path, where to save model')
    parser.add_argument('--model_load', type=str, default='', help='Path of existing model to load. '
                                                                   'If empty creates new model')
    parser.add_argument('--load_multi', type=bool, default=False, help='Load multi gpu model saved during training')
    parser.add_argument('--shuffle', type=bool, default=True, help='Toggle shuffle batches after epoch')
    parser.add_argument('--dropout', type=float, default=0.2, help='Set dropout value')
    parser.add_argument('--checkpoint', type=int, default=10, help='No. of epochs before save during training')
    parser.add_argument('--num_gpu', type=int, default=1, help='No. of gpu for multi gpu training. Must be even number')

    args = parser.parse_args()

    main(args)
