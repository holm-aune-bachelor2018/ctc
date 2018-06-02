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

import numpy as np

from DataGenerator import DataGenerator
from data import combine_all_wavs_and_trans_from_csvs
from utils.feature_utils import load_audio, convert_and_pad_transcripts, extract_mfcc_and_pad, \
    extract_mel_spectrogram_and_pad


class TestDataGen(unittest.TestCase):
    def setUp(self):
        _, self.df = combine_all_wavs_and_trans_from_csvs("data_dir/sample_librivox-test-clean.csv")

        self.dg = DataGenerator(self.df, batch_size=10, epoch_length=10)

    def tearDown(self):
        del self.dg

    # Data generator
    def test_extract_features_and_pad(self):
        indexes = np.arange(5)
        x_data_raw, y_data_raw, sr = load_audio(self.df, indexes_in_batch=indexes)
        x_data, input_length = self.dg.extract_features_and_pad(x_data_raw, sr)

        self.assertEqual(x_data.shape, (5,382,26))
        self.assertEqual(len(input_length), 5)
        self.assertLessEqual(all(input_length), 382)

    def test_get_seq_size(self):
        x_data_raw, _, sr = load_audio(self.df, indexes_in_batch=[0])

        size = self.dg.get_seq_size(x_data_raw[0], sr)

        self.assertEqual(size, 256)

    def test_get_item(self):
        batch0, _ = self.dg.__getitem__(0)
        batch1, _ = self.dg.__getitem__(1)

        x_data0 = batch0.get("the_input")
        x_data1 = batch1.get("the_input")
        y_data0 = batch0.get("the_labels")
        y_data1 = batch1.get("the_labels")
        input_length = batch0.get("input_length")
        label_length = batch0.get("label_length")

        self.assertTupleEqual(x_data0.shape, (10, 494, 26))
        self.assertTupleEqual(x_data1.shape, (9, 1514, 26))
        self.assertEqual(y_data0.shape[0], 10)
        self.assertEqual(y_data1.shape[0], 9)
        self.assertEqual(input_length.shape[0], 10)
        self.assertEqual(label_length.shape[0], 10)

    # Feature generation utils
    def test_load_audio(self):
        indexes = np.arange(5)
        x_data_raw, y_data_raw, sr = load_audio(self.df, indexes_in_batch=indexes)

        self.assertEqual(len(x_data_raw), 5)
        self.assertEqual(len(y_data_raw), 5)

    def test_extract_mfcc(self):
        x_data_raw, _, sr = load_audio(self.df, indexes_in_batch=[0])

        mfcc_padded, x_length = extract_mfcc_and_pad(x_data_raw[0], sr=sr, max_pad_length=500, frame_length=320,
                                                     hop_length=160, mfcc_features=26, n_mels=40)

        self.assertTupleEqual(mfcc_padded.shape, (500, 26))
        self.assertEqual(x_length, 256)

    def test_extract_mel_spec(self):
        x_data_raw, _, sr = load_audio(self.df, indexes_in_batch=[0])
        mel_spec, x_length = extract_mel_spectrogram_and_pad(x_data_raw[0], sr=sr, max_pad_length=500,
                                                             frame_length=320, hop_length=160, n_mels=40)

        self.assertTupleEqual(mel_spec.shape, (500, 40))
        self.assertEqual(x_length, 256)

    def test_convert_transcripts(self):
        _, y_data_raw, sr = load_audio(self.df, indexes_in_batch=[0])
        transcript, y_length = convert_and_pad_transcripts(y_data_raw)
        exp = [23.,  5., 18.,  5.,  0.,  9.,  0.,  2., 21., 20.,  0.,  1., 12., 18.,  5.,  1.,  4., 25., 0.,
               15., 14.,  0., 20.,  8.,  5.,  0.,  3.,  1., 18., 20.]

        list = transcript[0].tolist()
        self.assertListEqual(list, exp)
        self.assertEqual(y_length, 30)


if __name__ == '__main__':
    unittest.main()
