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

# CTC implementation from Keras example found at https://github.com/keras-team/keras/blob/master/examples/image_ocr.py

from keras import backend as K
from keras.layers import Dense, SimpleRNN, LSTM, CuDNNLSTM, Bidirectional, TimeDistributed, Conv1D, ZeroPadding1D
from keras.layers import Lambda, Input, Dropout, Masking
from keras.models import Model


def model(model_type='brnn', units=512, input_dim=26, output_dim=29, dropout=0.2, cudnn=False, n_layers=1):
    if model_type == 'brnn':
        network_model = brnn(units, input_dim, output_dim, dropout)

    elif model_type == 'deep_rnn':
        network_model = deep_rnn(units, input_dim, output_dim, dropout, n_layers=n_layers)

    elif model_type == 'blstm':
        network_model = blstm(units, input_dim, output_dim, dropout, cudnn=cudnn, n_layers=n_layers)

    elif model_type == 'deep_lstm':
        network_model = deep_lstm(units, input_dim, output_dim, dropout, cudnn=cudnn, n_layers=n_layers)

    elif model_type == 'cnn_blstm':
        network_model = cnn_blstm(units, input_dim, output_dim, dropout, cudnn=cudnn, n_layers=n_layers)

    else:
        raise ValueError("Not a valid model: ", model_type)

    return network_model


# Architecture from Baidu Deep speech: Scaling up end-to-end speech recognition (https://arxiv.org/pdf/1412.5567.pdf)
def brnn(units, input_dim=26, output_dim=29, dropout=0.2, numb_of_dense=3, n_layers=1):
    """
    :param units: Hidden units per layer
    :param input_dim: Size of input dimension (number of features), default=26
    :param output_dim: Output dim of final layer of model (input to CTC layer), default=29
    :param dropout: Dropout rate, default=0.2
    :param numb_of_dense: Number of fully connected layers before recurrent, default=3
    :param n_layers: Number of bidirectional recurrent layers, default=1
    :return: network_model: brnn

    Default model contains:
     1 layer of masking
     3 layers of fully connected clipped ReLu (DNN) with dropout 20 % between each layer
     1 layer of BRNN
     1 layers of fully connected clipped ReLu (DNN) with dropout 20 % between each layer
     1 layer of softmax
    """

    # Input data type
    dtype = 'float32'
    # Kernel and bias initializers for fully connected dense layers
    kernel_init_dense = 'random_normal'
    bias_init_dense = 'random_normal'

    # Kernel and bias initializers for recurrent layer
    kernel_init_rnn = 'glorot_uniform'
    bias_init_rnn = 'zeros'

    # ---- Network model ----
    # x_input layer, dim: (batch_size * x_seq_size * features)
    input_data = Input(name='the_input',shape=(None, input_dim), dtype=dtype)

    # Masking layer
    x = Masking(mask_value=0., name='masking')(input_data)

    # Default 3 fully connected layers DNN ReLu
    # Default dropout rate 20 % at each FC layer
    for i in range(0, numb_of_dense):
        x = TimeDistributed(Dense(units=units, kernel_initializer=kernel_init_dense, bias_initializer=bias_init_dense,
                                  activation=clipped_relu), name='fc_'+str(i+1))(x)
        x = TimeDistributed(Dropout(dropout), name='dropout_'+str(i+1))(x)

    # Bidirectional RNN (with ReLu)
    for i in range(0, n_layers):
        x = Bidirectional(SimpleRNN(units, activation='relu', kernel_initializer=kernel_init_rnn, dropout=0.2,
                                    bias_initializer=bias_init_rnn, return_sequences=True),
                          merge_mode='concat', name='bi_rnn'+str(i+1))(x)

    # 1 fully connected layer DNN ReLu with default 20% dropout
    x = TimeDistributed(Dense(units=units, kernel_initializer=kernel_init_dense, bias_initializer=bias_init_dense,
                              activation='relu'), name='fc_4')(x)
    x = TimeDistributed(Dropout(dropout), name='dropout_4')(x)

    # Output layer with softmax
    y_pred = TimeDistributed(Dense(units=output_dim, kernel_initializer=kernel_init_dense,
                                   bias_initializer=bias_init_dense, activation='softmax'), name='softmax')(x)

    # ---- CTC ----
    # y_input layers (transcription data) for CTC loss
    labels = Input(name='the_labels', shape=[None], dtype=dtype)        # transcription data (batch_size * y_seq_size)
    input_length = Input(name='input_length', shape=[1], dtype=dtype)   # unpadded len of all x_sequences in batch
    label_length = Input(name='label_length', shape=[1], dtype=dtype)   # unpadded len of all y_sequences in batch

    # Lambda layer with ctc_loss function due to Keras not supporting CTC layers
    loss_out = Lambda(function=ctc_lambda_func, name='ctc', output_shape=(1,))(
                      [y_pred, labels, input_length, label_length])

    network_model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

    return network_model


def deep_rnn(units, input_dim=26, output_dim=29, dropout=0.2, numb_of_dense=3, n_layers=3):
    """
    :param units: Hidden units per layer
    :param input_dim: Size of input dimension (number of features), default=26
    :param output_dim: Output dim of final layer of model (input to CTC layer), default=29
    :param dropout: Dropout rate, default=0.2
    :param numb_of_dense: Number of fully connected layers before recurrent, default=3
    :param n_layers: Number of simple RNN layers, default=3
    :return: network_model: deep_rnn

    Default model contains:
     1 layer of masking
     3 layers of fully connected clipped ReLu (DNN) with dropout 20 % between each layer
     3 layers of RNN with 20% dropout
     1 layers of fully connected clipped ReLu (DNN) with dropout 20 % between each layer
     1 layer of softmax
    """

    # Input data type
    dtype = 'float32'

    # Kernel and bias initializers for fully connected dense layers
    kernel_init_dense = 'random_normal'
    bias_init_dense = 'random_normal'

    # Kernel and bias initializers for recurrent layer
    kernel_init_rnn = 'glorot_uniform'
    bias_init_rnn = 'zeros'

    # ---- Network model ----
    # x_input layer, dim: (batch_size * x_seq_size * mfcc_features)
    input_data = Input(name='the_input',shape=(None, input_dim), dtype=dtype)

    # Masking layer
    x = Masking(mask_value=0., name='masking')(input_data)

    # Default 3 fully connected layers DNN ReLu
    # Default dropout rate 20 % at each FC layer
    for i in range(0, numb_of_dense):
        x = TimeDistributed(Dense(units=units, kernel_initializer=kernel_init_dense, bias_initializer=bias_init_dense,
                                  activation=clipped_relu), name='fc_'+str(i+1))(x)
        x = TimeDistributed(Dropout(dropout), name='dropout_'+str(i+1))(x)

    # Deep RNN network with a default of 3 layers
    for i in range(0, n_layers):
        x = SimpleRNN(units, activation='relu', kernel_initializer=kernel_init_rnn, bias_initializer=bias_init_rnn,
                      dropout=dropout, return_sequences=True, name=('deep_rnn_'+ str(i+1)))(x)

    # 1 fully connected layer DNN ReLu with default 20% dropout
    x = TimeDistributed(Dense(units=units, kernel_initializer=kernel_init_dense, bias_initializer=bias_init_dense,
                              activation='relu'), name='fc_4')(x)
    x = TimeDistributed(Dropout(dropout), name='dropout_4')(x)

    # Output layer with softmax
    y_pred = TimeDistributed(Dense(units=output_dim, kernel_initializer=kernel_init_dense,
                                   bias_initializer=bias_init_dense, activation='softmax'), name='softmax')(x)

    # ---- CTC ----
    # y_input layers (transcription data) for CTC loss
    labels = Input(name='the_labels', shape=[None], dtype=dtype)        # transcription data (batch_size * y_seq_size)
    input_length = Input(name='input_length', shape=[1], dtype=dtype)   # unpadded len of all x_sequences in batch
    label_length = Input(name='label_length', shape=[1], dtype=dtype)   # unpadded len of all y_sequences in batch

    # Lambda layer with ctc_loss function due to Keras not supporting CTC layers
    loss_out = Lambda(function=ctc_lambda_func, name='ctc', output_shape=(1,))(
                      [y_pred, labels, input_length, label_length])

    network_model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

    return network_model


def blstm(units, input_dim=26, output_dim=29, dropout=0.2, numb_of_dense=3, cudnn=False, n_layers=1):
    """
    :param units: Hidden units per layer
    :param input_dim: Size of input dimension (number of features), default=26
    :param output_dim: Output dim of final layer of model (input to CTC layer), default=29
    :param dropout: Dropout rate, default=0.2
    :param numb_of_dense: Number of fully connected layers before recurrent, default=3
    :param cudnn: Whether to use the CuDNN optimized LSTM (GPU only), default=False
    :param n_layers: Number of stacked BLSTM layers, default=1
    :return: network_model: blstm

    Default model contains:
     1 layer of masking
     3 layers of fully connected clipped ReLu (DNN) with dropout 20 % between each layer
     1 layer of BLSTM
     1 layers of fully connected clipped ReLu (DNN) with dropout 20 % between each layer
     1 layer of softmax
    """

    # Input data type
    dtype = 'float32'

    # Kernel and bias initializers for fully connected dense layers
    kernel_init_dense = 'random_normal'
    bias_init_dense = 'random_normal'

    # Kernel and bias initializers for recurrent layer
    kernel_init_rnn = 'glorot_uniform'
    bias_init_rnn = 'random_normal'

    # ---- Network model ----
    # x_input layer, dim: (batch_size * x_seq_size * features)
    input_data = Input(name='the_input', shape=(None, input_dim), dtype=dtype)

    if cudnn:
        # CuDNNLSTM does not support masking
        x = input_data
    else:
        # Masking layer
        x = Masking(mask_value=0., name='masking')(input_data)

    # Default 3 fully connected layers DNN ReLu
    # Default dropout rate 20 % at each FC layer
    for i in range(0, numb_of_dense):
        x = TimeDistributed(Dense(units=units, kernel_initializer=kernel_init_dense, bias_initializer=bias_init_dense,
                                  activation=clipped_relu), name='fc_'+str(i+1))(x)
        x = TimeDistributed(Dropout(dropout), name='dropout_'+str(i+1))(x)

    # Bidirectional RNN (with ReLu)
    # If running on GPU, use the CuDNN optimised LSTM model
    if cudnn:
        for i in range(0, n_layers):
            x = Bidirectional(CuDNNLSTM(units, kernel_initializer=kernel_init_rnn, bias_initializer=bias_init_rnn,
                                        unit_forget_bias=True, return_sequences=True),
                              merge_mode='sum', name=('CuDNN_bi_lstm' + str(i+1)))(x)
    else:
        for i in range(0, n_layers):
            x = Bidirectional(LSTM(units, activation='relu', kernel_initializer=kernel_init_rnn, dropout=dropout,
                                   bias_initializer=bias_init_rnn, return_sequences=True),
                              merge_mode='sum', name=('bi_lstm' + str(i+1)))(x)

    # 1 fully connected layer DNN ReLu with default 20% dropout
    x = TimeDistributed(Dense(units=units, kernel_initializer=kernel_init_dense, bias_initializer=bias_init_dense,
                              activation='relu'), name='fc_4')(x)
    x = TimeDistributed(Dropout(dropout), name='dropout_4')(x)

    # Output layer with softmax
    y_pred = TimeDistributed(Dense(units=output_dim, kernel_initializer=kernel_init_dense,
                                   bias_initializer=bias_init_dense, activation='softmax'), name='softmax')(x)

    # ---- CTC ----
    # y_input layers (transcription data) for CTC loss
    labels = Input(name='the_labels', shape=[None], dtype=dtype)       # transcription data (batch_size * y_seq_size)
    input_length = Input(name='input_length', shape=[1], dtype=dtype)  # unpadded len of all x_sequences in batch
    label_length = Input(name='label_length', shape=[1], dtype=dtype)  # unpadded len of all y_sequences in batch

    # Lambda layer with ctc_loss function due to Keras not supporting CTC layers
    loss_out = Lambda(function=ctc_lambda_func, name='ctc', output_shape=(1,))(
                      [y_pred, labels, input_length, label_length])

    network_model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

    return network_model


def deep_lstm(units, input_dim=26, output_dim=29, dropout=0.2, numb_of_dense=3, cudnn=False, n_layers=3):
    """
    :param units: Hidden units per layer
    :param input_dim: Size of input dimension (number of features), default=26
    :param output_dim: Output dim of final layer of model (input to CTC layer), default=29
    :param dropout: Dropout rate, default=0.2
    :param numb_of_dense: Number of fully connected layers before recurrent, default=3
    :param cudnn: Whether to use the CuDNN optimized LSTM (only for GPU), default=False
    :param n_layers: Number of LSTM layers, default=3
    :return: network_model: deep_lstm

    Default model contains:
     3 layers of fully connected clipped ReLu (DNN) with dropout 20 % between each layer
     3 layers LSTM
     1 layers of fully connected clipped ReLu (DNN) with dropout 20 % between each layer
     1 layer of softmax
    """

    # Input data type
    dtype = 'float32'

    # Kernel and bias initializers for fully connected dense layers
    kernel_init_dense = 'random_normal'
    bias_init_dense = 'random_normal'

    # Kernel and bias initializers for recurrent layer
    kernel_init_rnn = 'glorot_uniform'
    bias_init_rnn = 'random_normal'

    # ---- Network model ----
    # x_input layer, dim: (batch_size * x_seq_size * features)
    input_data = Input(name='the_input', shape=(None, input_dim), dtype=dtype)

    if cudnn:
        x = input_data
    else:
        x = Masking()(input_data)

    # Default 3 fully connected layers DNN ReLu
    # Default dropout rate 20 % at each FC layer
    for i in range(0, numb_of_dense):
        x = TimeDistributed(Dense(units=units, kernel_initializer=kernel_init_dense, bias_initializer=bias_init_dense,
                                  activation=clipped_relu), name='fc_'+str(i+1))(x)
        x = TimeDistributed(Dropout(dropout), name='dropout_'+str(i+1))(x)

    # Default 3 LSTM layers
    if cudnn:
        for i in range(0, n_layers):
            x = CuDNNLSTM(units, kernel_initializer=kernel_init_rnn, bias_initializer=bias_init_rnn,
                          unit_forget_bias=True, return_sequences=True, name='CuDNN_lstm'+str(i+1))(x)
    else:
        for i in range(0, n_layers):
            x = LSTM(units, activation='relu', kernel_initializer=kernel_init_rnn, bias_initializer=bias_init_rnn,
                     dropout=dropout, return_sequences=True, name='lstm'+str(i+1))(x)

    # 1 fully connected layer DNN ReLu with default 20% dropout
    x = TimeDistributed(Dense(units=units, kernel_initializer=kernel_init_dense, bias_initializer=bias_init_dense,
                              activation='relu'), name='fc_4')(x)
    x = TimeDistributed(Dropout(dropout), name='dropout_4')(x)

    # Output layer with softmax, default output_dim = 29 units
    y_pred = TimeDistributed(Dense(units=output_dim, kernel_initializer=kernel_init_dense,
                                   bias_initializer=bias_init_dense, activation='softmax'), name='softmax')(x)

    # ---- CTC ----
    # y_input layers (transcription data) for CTC loss
    labels = Input(name='the_labels', shape=[None], dtype=dtype)       # transcription data (batch_size * y_seq_size)
    input_length = Input(name='input_length', shape=[1], dtype=dtype)  # unpadded len of all x_sequences in batch
    label_length = Input(name='label_length', shape=[1], dtype=dtype)  # unpadded len of all y_sequences in batch

    # Lambda layer with ctc_loss function due to Keras not supporting CTC layers
    loss_out = Lambda(function=ctc_lambda_func, name='ctc', output_shape=(1,))(
                      [y_pred, labels, input_length, label_length])

    network_model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

    return network_model


def cnn_blstm(units, input_dim=26, output_dim=29, dropout=0.2, seq_padding=2176, cudnn=False, n_layers=1):
    """
    :param units: Hidden units per layer
    :param input_dim: Size of input dimension (number of features), default=26
    :param output_dim: Output dim of final layer of model (input to CTC layer), default=29
    :param dropout: Dropout rate, default=0.2
    :param seq_padding: length of sequence zero padding before conv layers, default=2176
    :param cudnn: Whether to use the CuDNN optimized LSTM (only for GPU), default=False
    :param n_layers: Number of stacked BLSTM layers, default=1
    :return: network_model: cnn_blstm

    Model contains:
     3 layers of CNN Conv1D
     3 layers of BLSTM
     1 layers of fully connected clipped ReLu (DNN) with dropout 20 % between each layer
     1 layer of softmax
    """

    # Input data type
    dtype = 'float32'

    activation_conv = clipped_relu

    # Kernel and bias initializers for fully connected dense layer
    kernel_init_dense = 'random_normal'
    bias_init_dense = 'random_normal'

    # Kernel and bias initializers for convolution layers
    kernel_init_conv = 'glorot_uniform'
    bias_init_conv = 'random_normal'

    # Kernel and bias initializers for recurrent layer
    kernel_init_rnn = 'glorot_uniform'
    bias_init_rnn = 'random_normal'

    # ---- Network model ----
    input_data = Input(name='the_input', shape=(None, input_dim), dtype=dtype)

    # Pad on sequence dim so all sequences are equal length
    x = ZeroPadding1D(padding=(0, seq_padding))(input_data)

    # 3 x 1D convolutional layers with strides: 1, 1, 2
    x = Conv1D(filters=units, kernel_size=5, strides=1, activation=activation_conv,
               kernel_initializer=kernel_init_conv, bias_initializer=bias_init_conv, name='conv_1')(x)
    x = TimeDistributed(Dropout(dropout), name='dropout_1')(x)

    x = Conv1D(filters=units, kernel_size=5, strides=1, activation=activation_conv,
               kernel_initializer=kernel_init_conv, bias_initializer=bias_init_conv, name='conv_2')(x)
    x = TimeDistributed(Dropout(dropout), name='dropout_2')(x)

    x = Conv1D(filters=units, kernel_size=5, strides=2, activation=activation_conv,
               kernel_initializer=kernel_init_conv, bias_initializer=bias_init_conv, name='conv_3')(x)
    x = TimeDistributed(Dropout(dropout), name='dropout_3')(x)

    # Bidirectional LSTM
    if cudnn:
        for i in range(0, n_layers):
            x = Bidirectional(CuDNNLSTM(units, kernel_initializer=kernel_init_rnn, bias_initializer=bias_init_rnn,
                                        unit_forget_bias=True, return_sequences=True),
                              merge_mode='sum', name='CuDNN_bi_lstm'+str(i+1))(x)
    else:
        for i in range(0, n_layers):
            x = Bidirectional(LSTM(units, activation='relu', kernel_initializer=kernel_init_rnn, dropout=dropout,
                                   bias_initializer=bias_init_rnn, return_sequences=True),
                              merge_mode='sum', name='bi_lstm'+str(i+1))(x)

    # 1 fully connected layer DNN ReLu with default 20% dropout
    x = TimeDistributed(Dense(units=units, kernel_initializer=kernel_init_dense, bias_initializer=bias_init_dense,
                              activation='relu'), name='fc_4')(x)
    x = TimeDistributed(Dropout(dropout), name='dropout_4')(x)

    # Output layer with softmax
    y_pred = TimeDistributed(Dense(units=output_dim, kernel_initializer=kernel_init_dense,
                                   bias_initializer=bias_init_dense, activation='softmax'), name='softmax')(x)

    # ---- CTC ----
    # y_input layers (transcription data) for CTC loss
    labels = Input(name='the_labels', shape=[None], dtype=dtype)       # transcription data (batch_size * y_seq_size)
    input_length = Input(name='input_length', shape=[1], dtype=dtype)  # unpadded len of all x_sequences in batch
    label_length = Input(name='label_length', shape=[1], dtype=dtype)  # unpadded len of all y_sequences in batch

    # Lambda layer with ctc_loss function due to Keras not supporting CTC layers
    loss_out = Lambda(function=ctc_lambda_func, name='ctc', output_shape=(1,))(
                      [y_pred, labels, input_length, label_length])

    network_model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

    return network_model


# Lambda implementation of CTC loss, using ctc_batch_cost from TensorFlow backend
# CTC implementation from Keras example found at https://github.com/keras-team/keras/blob/master/examples/image_ocr.py
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    # print "y_pred_shape: ", y_pred.shape
    y_pred = y_pred[:, 2:, :]
    # print "y_pred_shape: ", y_pred.shape
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


# Returns clipped relu, clip value set to 20.
def clipped_relu(value):
    return K.relu(value, max_value=20)
