"""
LICENSE

This file is part of Speech recognition with CTC in Keras.
The project is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
version.
The project is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this project.
If not, see http://www.gnu.org/licenses/.

"""

import argparse

import keras.backend as K
from keras import models

import models
from DataGenerator import DataGenerator
from data import combine_all_wavs_and_trans_from_csvs
from utils.train_utils import predict_on_batch, calc_wer


def main(args):
    try:
        if not args.model_load:
            raise ValueError("Error within model arguments")
        audio_dir = args.audio_dir

        print("\nReading test data: ")
        _, df = combine_all_wavs_and_trans_from_csvs(audio_dir)

        batch_size = args.batch_size
        batch_index = args.batch_index

        mfcc_features = args.mfccs
        n_mels = args.mels
        # Sampling rate of data in khz (LibriSpeech is 16khz)
        frequency = 16

        # Training data_params:
        model_load = args.model_load
        load_multi = args.load_multi

        # Sets the full dataset in audio_dir to be available through data_generator
        # The data_generator doesn't actually load the audio files until they are requested through __get_item__()
        epoch_length = 0

        # Load trained model
        # When loading custom objects, Keras needs to know where to find them.
        # The CTC lambda is a dummy function
        custom_objects = {'clipped_relu': models.clipped_relu,
                          '<lambda>': lambda y_true, y_pred: y_pred}

        # When loading a parallel model saved *while* running on GPU, use load_multi
        if load_multi:
            model = models.load_model(
                model_load, custom_objects=custom_objects)
            model = model.layers[-2]
            print("\nLoaded existing model: ", model_load)

        # Load single GPU/CPU model or model saved *after* finished training
        else:
            model = models.load_model(
                model_load, custom_objects=custom_objects)
            print("\nLoaded existing model: ", model_load)

        # Dummy loss-function to compile model, actual CTC loss-function defined as a lambda layer in model
        loss = {'ctc': lambda y_true, y_pred: y_pred}

        model.compile(loss=loss, optimizer='Adam')

        feature_shape = model.input_shape[0][2]

        # Model feature type
        if not args.feature_type:
            if feature_shape == 26:
                feature_type = 'mfcc'
            else:
                feature_type = 'spectrogram'
        else:
            feature_type = args.feature_type

        print("Feature type: ", feature_type)

        # Data generation parameters
        data_params = {'feature_type': feature_type,
                       'batch_size': batch_size,
                       'frame_length': 20 * frequency,
                       'hop_length': 10 * frequency,
                       'mfcc_features': mfcc_features,
                       'n_mels': n_mels,
                       'epoch_length': epoch_length,
                       'shuffle': False
                       }

        # Data generators for training, validation and testing data
        data_generator = DataGenerator(df, **data_params)

        # Print(model summary)
        model.summary()

        # Creates a test function that takes preprocessed sound input and outputs predictions
        # Used to calculate WER while training the network
        input_data = model.get_layer('the_input').input
        y_pred = model.get_layer('ctc').input[0]
        test_func = K.function([input_data], [y_pred])

        if args.calc_wer:
            print("\n - Calculation WER on ", audio_dir)
            wer = calc_wer(test_func, data_generator)
            print("Average WER: ", wer[1])

        predictions = predict_on_batch(data_generator, test_func, batch_index)
        print("\n - Predictions from batch index: ",
              batch_index, "\nFrom: ", audio_dir, "\n")
        for i in predictions:
            print("Original: ", i[0])
            print("Predicted: ", i[1], "\n")

    except (Exception, BaseException, GeneratorExit, SystemExit) as e:
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(e).__name__, e.args)
        print("e.args: ", e.args)
        print(message)

    finally:
        # Clear memory
        K.clear_session()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Predict data params:
    parser.add_argument('--audio_dir', type=str, default="data_dir/librivox-test-clean.csv",
                        help='Path to .csv file of audio to predict')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Number of files to predict.')
    parser.add_argument('--batch_index', type=int, default=10,
                        help='Index of batch in sorted .csv file to predict.')
    parser.add_argument('--calc_wer', action='store_true',
                        help='Calculate the word error rate on the data in audio_dir.')

    # Only need to specify these if feature params are changed from default (different than 26 MFCC and 40 mels)
    parser.add_argument('--feature_type', type=str,
                        help='Feature extraction method: mfcc or spectrogram. '
                        'If none is specified it tries to detect feature type from input_shape.')
    parser.add_argument('--mfccs', type=int, default=26,
                        help='Number of mfcc features per frame to extract.')
    parser.add_argument('--mels', type=int, default=40,
                        help='Number of mels to use in feature extraction.')

    # Model load params:
    parser.add_argument('--model_load', type=str,
                        help='Path of existing model to load.')
    parser.add_argument('--load_multi', action='store_true',
                        help='Load multi gpu model saved during parallel GPU training.')

    args = parser.parse_args()

    main(args)
