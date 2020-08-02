import argparse
from datetime import datetime

import keras.backend as K
import tensorflow as tf
from keras import models
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from keras.utils import multi_gpu_model

import models
from DataGenerator import DataGenerator
from LossCallback import LossCallback
from data import combine_all_wavs_and_trans_from_csvs

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main(args):
    # Create dataframes
    print("\nReading training data:")
    _, input_dataframe = combine_all_wavs_and_trans_from_csvs(args.trainDataset)
    print("\nReading validation data: ")
    _, validation_df = combine_all_wavs_and_trans_from_csvs(args.validationDataset)
    print("\nReading test data: ")
    _, test_df = combine_all_wavs_and_trans_from_csvs(args.testDataset)

    # Training params:
    batch_size = args.batch_size
    input_epoch_length = args.epoch_len
    epochs = args.epochs
    learning_rate = args.lr
    log_file = args.log_file

    # Multi GPU or single GPU / CPU training
    num_gpu = args.num_gpu

    # Preprocessing params
    feature_type = args.feature_type
    mfcc_features = args.mfccs
    n_mels = args.mels

    # Model params
    model_type = args.model_type
    units = args.units
    dropout = args.dropout
    n_layers = args.layers

    # Saving and loading params
    model_save = args.model_save
    checkpoint = args.checkpoint
    model_load = args.model_load
    load_multi = args.load_multi

    # Additional settings for training
    # Save model with best val_loss (on path "model_save" + "_best")
    save_best = args.save_best_val
    shuffle = args.shuffle_indexes
    reduce_lr = args.reduce_lr              # Reduce learning rate on val_loss plateau
    # Stop training early if val_loss stops improving
    early_stopping = args.early_stopping

    # Sampling rate of data in khz (LibriSpeech is 16khz)
    frequency = 16
    cudnnlstm = False

    # Data generation parameters
    data_params = {'feature_type': feature_type,
                   'batch_size': batch_size,
                   'frame_length': 20 * frequency,
                   'hop_length': 10 * frequency,
                   'mfcc_features': mfcc_features,
                   'n_mels': n_mels,
                   'epoch_length': input_epoch_length,
                   'shuffle': shuffle
                   }

    # Data generators for training, validation and testing data
    training_generator = DataGenerator(input_dataframe, **data_params)
    validation_generator = DataGenerator(validation_df, **data_params)
    test_generator = DataGenerator(test_df, **data_params)

    # Model input shape
    if feature_type == 'mfcc':
        input_dim = mfcc_features
    else:
        input_dim = n_mels

    output_dim = 29  # Output dim: features to predict + 1 for the CTC blank prediction

    # Optimization algorithm used to update network weights
    optimizer = Adam(lr=learning_rate, epsilon=1e-8, clipnorm=2.0)

    # Dummy loss-function for compiling model, actual CTC loss-function defined as a lambda layer in model
    loss = {'ctc': lambda y_true, y_pred: y_pred}

    # Print(training data at the beginning of training)
    calc_epoch_length = training_generator.__len__()
    print("\n\nModel and training parameters: ")
    print("Starting time: ", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print(" - epochs: ", epochs, "\n - batch size: ", batch_size,
          "\n - input epoch length: ", input_epoch_length, "\n - network epoch length: ", calc_epoch_length,
          "\n - training on ", calc_epoch_length *
          batch_size, " files", "\n - learning rate: ", learning_rate,
          "\n - hidden units: ", units, "\n - mfcc features: ", mfcc_features, "\n - dropout: ", dropout, "\n")

    try:
        # Load previous model or create new. With device cpu ensures that the model is created/loaded on the cpu
        if model_load:
            with tf.device('/cpu:0'):
                # When loading custom objects, Keras needs to know where to find them.
                # The CTC lambda is a dummy function
                custom_objects = {'clipped_relu': models.clipped_relu,
                                  '<lambda>': lambda y_true, y_pred: y_pred}

                # When loading a parallel model saved *while* running on multiple GPUs, use load_multi
                if load_multi:
                    model = models.load_model(
                        model_load, custom_objects=custom_objects)
                    model = model.layers[-2]
                    print("Loaded existing model at: ", model_load)

                # Load single GPU/CPU model or model saved *after* finished training
                else:
                    model = models.load_model(
                        model_load, custom_objects=custom_objects)
                    print("Loaded existing model at: ", model_load)

        else:
            with tf.device('/cpu:0'):
                # Create new model
                model = models.model(model_type=model_type, units=units, input_dim=input_dim,
                                     output_dim=output_dim, dropout=dropout, cudnn=cudnnlstm, n_layers=n_layers)
                print("Creating new model: ", model_type)

        # Loss callback parameters
        loss_callback_params = {'validation_gen': validation_generator,
                                'test_gen': test_generator,
                                'checkpoint': checkpoint,
                                'path_to_save': model_save,
                                'log_file_path': log_file
                                }

        # Model training parameters
        model_train_params = {'generator': training_generator,
                              'epochs': epochs,
                              'verbose': 2,
                              'validation_data': validation_generator,
                              'workers': 1,
                              'shuffle': shuffle}

        # Optional callbacks for added functionality
        # Reduces learning rate when val_loss stagnates.
        if reduce_lr:
            print("Reducing learning rate on plateau")
            reduce_lr_cb = ReduceLROnPlateau(
                factor=0.2, patience=5, verbose=0, epsilon=0.1, min_lr=0.0000001)
            callbacks = [reduce_lr_cb]
        else:
            callbacks = []

        # Stops the model early if the val_loss isn't improving
        if early_stopping:
            es_cb = EarlyStopping(min_delta=0, patience=5,
                                  verbose=0, mode='auto')
            callbacks.append(es_cb)

        # Saves the model if val_loss is improved at "model_save" + "_best"
        if save_best:
            save_best = model_save + str('_best')
            mcp_cb = ModelCheckpoint(
                save_best, verbose=1, save_best_only=True, period=1)
            callbacks.append(mcp_cb)

        # Train with CPU or single GPU
        elif num_gpu == 1 or num_gpu == 0:
            # Compile model for training on GPUs < 2
            model.compile(loss=loss, optimizer=optimizer)

            # Print(model summary)
            model.summary()

            # Creates a test function that takes preprocessed sound input and outputs predictions
            # Used to calculate WER while training the network
            input_data = model.get_layer('the_input').input
            y_pred = model.get_layer('ctc').input[0]
            test_func = K.function([input_data], [y_pred])

            # The loss callback function that calculates WER while training
            loss_cb = LossCallback(test_func=test_func,
                                   model=model, **loss_callback_params)
            callbacks.append(loss_cb)

            # Run training
            model.fit_generator(callbacks=callbacks, **model_train_params)

        else:
            raise ValueError('Not a valid number of GPUs: ', num_gpu)

        if args.model_save:
            model.save(model_save)
            print("Model saved: ", model_save)

    except (Exception, ArithmeticError) as e:
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(e).__name__, e.args)
        print(message)

    finally:
        # Clear memory
        K.clear_session()
    print("Ending time: ", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Path params:
    parser.add_argument('-tr', '--trainDataset', metavar='trainDataset', dest='trainDataset',
                    help='Train dataset csv path', default='data/local/librivox-train-clean-1-wav.csv')
    parser.add_argument('-v', '--validationDataset', metavar='validationDataset', dest='validationDataset',
                            help='Validation dataset csv path', default='data/local/librivox-dev-clean-wav.csv')
    parser.add_argument('-te', '--testDataset', metavar='testDataset', dest='testDataset',
                            help='Test dataset csv path', default='data/local/librivox-test-clean-wav.csv')

    # Training params:
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Number of files in one batch.')
    parser.add_argument('--epoch_len', type=int, default=32,
                        help='Number of batches per epoch. 0 trains on full dataset.')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate.')
    parser.add_argument('--log_file', type=str, default="log",
                        help='Path to log stats to .csv file.')

    # Multi GPU or single GPU / CPU training
    parser.add_argument('--num_gpu', type=int, default=1,
                        help='No. of gpu for training. (0,1) sets up normal training, for CPU or one GPU. '
                             'MultiGPU training must be an even number larger than 1.')

    # Preprocessing params
    parser.add_argument('--feature_type', type=str, default='mfcc',
                        help='Feature extraction method: mfcc or spectrogram.')
    parser.add_argument('--mfccs', type=int, default=26,
                        help='Number of mfcc features per frame to extract.')
    parser.add_argument('--mels', type=int, default=40,
                        help='Number of mels to use in feature extraction.')

    # Model params
    parser.add_argument('--model_type', type=str, default='brnn',
                        help='Model to train: brnn, blstm, deep_rnn, deep_lstm, cnn_blstm.')
    parser.add_argument('--units', type=int, default=256,
                        help='Number of hidden nodes.')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Set dropout value (0-1).')
    parser.add_argument('--layers', type=int, default=1,
                        help='Number of recurrent or deep layers.')
    parser.add_argument('--cudnn', action='store_true',
                        help='Whether to use cudnn optimized LSTM')

    # Saving and loading model params:
    parser.add_argument('--model_save', type=str,
                        help='Path, where to save model.')
    parser.add_argument('--checkpoint', type=int, default=10,
                        help='No. of epochs before save during training.')
    parser.add_argument('--model_load', type=str, default='',
                        help='Path of existing model to load. If empty creates new model.')
    parser.add_argument('--load_multi', action='store_true',
                        help='Load multi gpu model saved during parallel GPU training.')

    # Additional training settings
    parser.add_argument('--save_best_val', action='store_true',
                        help='Save additional version of model if val_loss improves.')
    parser.add_argument('--shuffle_indexes', action='store_true',
                        help='Shuffle batches after each epoch.')
    parser.add_argument('--reduce_lr', action='store_true',
                        help='Reduce the learning rate if model stops improving val_loss.')
    parser.add_argument('--early_stopping', action='store_true',
                        help='Stop the training early if val_loss stops improving.')

    args = parser.parse_args()

    main(args)
