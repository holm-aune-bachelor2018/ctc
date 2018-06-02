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

import unittest

import keras.backend as K
from keras.optimizers import Adam

import models
from DataGenerator import DataGenerator
from data import combine_all_wavs_and_trans_from_csvs


class TestModelCompile(unittest.TestCase):
    def setUp(self):
        self.optimizer = Adam(lr=0.0001, epsilon=1e-8, clipnorm=2.0)
        self.loss = {'ctc': lambda y_true, y_pred: y_pred}

    def tearDown(self):
        K.clear_session()

    def test_brnn_compile(self):
        try:
            model = models.brnn(units=256, input_dim=26, output_dim=29, dropout=0.2, numb_of_dense=3)
            model.compile(loss=self.loss, optimizer=self.optimizer)
        except ():
            model = None

        self.assertIsNotNone(model)

    def test_deep_rnn_compile(self):
        try:
            model = models.deep_rnn(units=256, input_dim=26, output_dim=29, dropout=0.2, numb_of_dense=3, n_layers=3)
            model.compile(loss=self.loss, optimizer=self.optimizer)
        except ():
            model = None

        self.assertIsNotNone(model)

    def test_blstm_compile(self):
        try:
            model = models.blstm(units=256, input_dim=26, output_dim=29, dropout=0.2, numb_of_dense=3, cudnn=False,
                                 n_layers=1)
            model.compile(loss=self.loss, optimizer=self.optimizer)
        except ():
            model = None

        self.assertIsNotNone(model)

    def test_deep_lstm_compile(self):
        try:
            model = models.deep_lstm(units=256, input_dim=26, output_dim=29, dropout=0.2, numb_of_dense=3,
                                     cudnn=False, n_layers=3)
            model.compile(loss=self.loss, optimizer=self.optimizer)
        except ():
            model = None

        self.assertIsNotNone(model)

    def test_cnn_blstm_compile(self):
        try:
            model = models.cnn_blstm(units=256, input_dim=26, output_dim=29, dropout=0.2, seq_padding=2048,
                                     cudnn=False, n_layers=1)
            model.compile(loss=self.loss, optimizer=self.optimizer)
        except ():
            model = None

        self.assertIsNotNone(model)

    def test_brnn_fit(self):
        sample_data="data_dir/sample_librivox-test-clean.csv"
        _, df = combine_all_wavs_and_trans_from_csvs(sample_data)
        data_generator = DataGenerator(df, feature_type='mfcc', batch_size=6, frame_length=320, hop_length=160,
                                       n_mels=40, mfcc_features=26, epoch_length=0, shuffle=True)
        model = models.brnn(units=256, input_dim=26, output_dim=29, dropout=0.2, numb_of_dense=3)
        model.compile(loss=self.loss, optimizer=self.optimizer)

        # Run training
        model.fit_generator(generator=data_generator, epochs=1, verbose=0)

    def test_cnn_blstm_fit(self):
        sample_data="data_dir/sample_librivox-test-clean.csv"
        _, df = combine_all_wavs_and_trans_from_csvs(sample_data)
        data_generator = DataGenerator(df, feature_type='spectrogram', batch_size=6, frame_length=320, hop_length=160,
                                       n_mels=40, epoch_length=0, shuffle=True)
        model = models.cnn_blstm(units=256, input_dim=40, output_dim=29, dropout=0.2, cudnn=False, n_layers=1)
        model.compile(loss=self.loss, optimizer=self.optimizer)

        # Run training
        model.fit_generator(generator=data_generator, epochs=1, verbose=0)


if __name__ == '__main__':
    unittest.main()
