from keras.models import Model
from keras.layers import Dense, SimpleRNN, LSTM, CuDNNLSTM, Bidirectional, Masking, TimeDistributed,\
    Lambda, Input, Dropout
from keras import backend as K


def model(model_type='default', units=512, input_dim=26, output_dim=29, dropout=0.2):

    if model_type == 'default' or model_type == 'dnn_brnn':
        network_model = dnn_brnn(units, input_dim, output_dim, dropout)

    elif model_type == 'dnn_blstm':
        network_model = dnn_blstm(units, input_dim, output_dim, dropout)

    elif model_type == 'deep_rnn':
        network_model = deep_rnn(units, input_dim, output_dim, dropout)

    else:
        print "Not a valid model."
        return 0

    return network_model


# Architecture from Baidu Deep speech 1
def dnn_brnn(units, input_dim=26, output_dim=29, dropout=0.2):
    """
    :param units: units
    :param input_dim: input_dim(mfcc_features)
    :param output_dim output dim of final layer of model, input to CTC layer
    :return: dnn_brnn model

    Model contains:
     1 layer of masking
     3 layers of fully connected clipped ReLu (DNN) with dropout 20 % between each layer
     1 layer of BRNN
     1 layers of fully connected clipped ReLu (DNN) with dropout 20 % between each layer
     1 layer of softmax
    """

    dtype = 'float32'
    # kernel and bias initializers for fully connected dense layers
    kernel_init_dense = 'random_normal'
    bias_init_dense = 'random_normal'

    # kernel and bias initializers for recurrent layer
    kernel_init_rnn = 'glorot_uniform'
    bias_init_rnn = 'zeros'

    # ---- Network model ----
    # x_input layer, dim: (batch_size * x_seq_size * mfcc_features)
    input_data = Input(name='the_input',shape=(None, input_dim), dtype=dtype)

    # Masking layer
    x = Masking(mask_value=0., name='masking')(input_data)

    # 3 fully connected layers DNN ReLu
    # Dropout rate 20 % at each FC layer
    x = TimeDistributed(Dense(units=units, kernel_initializer=kernel_init_dense,  bias_initializer=bias_init_dense,
                              activation=clipped_relu), name='fc_1')(x)
    x = TimeDistributed(Dropout(dropout), name='dropout_1')(x)

    x = TimeDistributed(Dense(units=units, kernel_initializer=kernel_init_dense, bias_initializer=bias_init_dense,
                              activation=clipped_relu), name='fc_2')(x)
    x = TimeDistributed(Dropout(dropout), name='dropout_2')(x)

    x = TimeDistributed(Dense(units=units, kernel_initializer=kernel_init_dense, bias_initializer=bias_init_dense,
                              activation=clipped_relu), name='fc_3')(x)
    x = TimeDistributed(Dropout(dropout), name='dropout_3')(x)

    # Bidirectional RNN (with ReLu)
    x = Bidirectional(SimpleRNN(units, activation='relu', kernel_initializer=kernel_init_rnn, dropout=0.2,
                                bias_initializer=bias_init_rnn, return_sequences=True),
                      merge_mode='concat', name='bi_rnn')(x)

    # 1 fully connected relu layer
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
    loss_out = Lambda(function=ctc_lambda_func, name='ctc', output_shape=(1,))\
                     ([y_pred, labels, input_length, label_length])

    network_model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

    return network_model


def deep_rnn(units, input_dim=26, output_dim=29, dropout=0.2, numb_of_rnn=3):
    """
    :param units: units
    :param input_dim: input_dim(mfcc_features)
    :param output_dim output dim of final layer of model, input to CTC layer
    :return: dnn_brnn model

    Model contains:
     1 layer of masking
     3 layers of fully connected clipped ReLu (DNN) with dropout 20 % between each layer
     1 layer of BRNN
     1 layers of fully connected clipped ReLu (DNN) with dropout 20 % between each layer
     1 layer of softmax
    """

    dtype = 'float32'
    # kernel and bias initializers for fully connected dense layers
    kernel_init_dense = 'random_normal'
    bias_init_dense = 'random_normal'

    # kernel and bias initializers for recurrent layer
    kernel_init_rnn = 'glorot_uniform'
    bias_init_rnn = 'zeros'

    # ---- Network model ----
    # x_input layer, dim: (batch_size * x_seq_size * mfcc_features)
    input_data = Input(name='the_input',shape=(None, input_dim), dtype=dtype)

    # Masking layer
    x = Masking(mask_value=0., name='masking')(input_data)

    # 3 fully connected layers DNN ReLu
    # Dropout rate 20 % at each FC layer
    x = TimeDistributed(Dense(units=units, kernel_initializer=kernel_init_dense,  bias_initializer=bias_init_dense,
                              activation=clipped_relu), name='fc_1')(x)
    x = TimeDistributed(Dropout(dropout), name='dropout_1')(x)

    x = TimeDistributed(Dense(units=units, kernel_initializer=kernel_init_dense, bias_initializer=bias_init_dense,
                              activation=clipped_relu), name='fc_2')(x)
    x = TimeDistributed(Dropout(dropout), name='dropout_2')(x)

    x = TimeDistributed(Dense(units=units, kernel_initializer=kernel_init_dense, bias_initializer=bias_init_dense,
                              activation=clipped_relu), name='fc_3')(x)
    x = TimeDistributed(Dropout(dropout), name='dropout_3')(x)

    # Deep RNN network with a default of 3 layers
    for i in range(0, numb_of_rnn):
        x = SimpleRNN(units, activation='relu', kernel_initializer=kernel_init_rnn, bias_initializer=bias_init_rnn,
                      dropout=dropout, return_sequences=True, name=('deep_rnn_'+ str(i+1)))(x)

    # 1 fully connected relu layer
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
    loss_out = Lambda(function=ctc_lambda_func, name='ctc', output_shape=(1,))\
                     ([y_pred, labels, input_length, label_length])

    network_model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

    return network_model


def dnn_blstm(units, input_dim=26, output_dim=29, dropout=0.2):
    """
        :param units: units
        :param input_dim: input_dim(mfcc_features)
        :param output_dim output dim of final layer of model, input to CTC layer
        :return: dnn_brnn model

        Model contains:
         1 layer of masking
         3 layers of fully connected clipped ReLu (DNN) with dropout 20 % between each layer
         1 layer of BLSTM
         1 layers of fully connected clipped ReLu (DNN) with dropout 20 % between each layer
         1 layer of softmax
        """

    dtype = 'float32'
    cudnn = False
    # kernel and bias initializers for fully connected dense layers
    kernel_init_dense = 'random_normal'
    bias_init_dense = 'random_normal'

    # kernel and bias initializers for recurrent layer
    kernel_init_rnn = 'glorot_uniform'
    bias_init_rnn = 'zeros'

    # ---- Network model ----
    # x_input layer, dim: (batch_size * x_seq_size * mfcc_features)
    input_data = Input(name='the_input', shape=(None, input_dim), dtype=dtype)

    # Masking layer
    x = Masking(mask_value=0., name='masking')(input_data)

    # 3 fully connected layers DNN ReLu
    # Dropout rate 20 % at each FC layer
    x = TimeDistributed(Dense(units=units, kernel_initializer=kernel_init_dense, bias_initializer=bias_init_dense,
                              activation=clipped_relu), name='fc_1')(x)
    x = TimeDistributed(Dropout(dropout), name='dropout_1')(x)

    x = TimeDistributed(Dense(units=units, kernel_initializer=kernel_init_dense, bias_initializer=bias_init_dense,
                              activation=clipped_relu), name='fc_2')(x)
    x = TimeDistributed(Dropout(dropout), name='dropout_2')(x)

    x = TimeDistributed(Dense(units=units, kernel_initializer=kernel_init_dense, bias_initializer=bias_init_dense,
                              activation=clipped_relu), name='fc_3')(x)
    x = TimeDistributed(Dropout(dropout), name='dropout_3')(x)

    # Bidirectional RNN (with ReLu)
    # If running on GPU, use the CuDNN optimised LSTM model
    if cudnn:
        x = Bidirectional(CuDNNLSTM(units, kernel_initializer=kernel_init_rnn,
                                    bias_initializer=bias_init_rnn, unit_forget_bias=True, return_sequences=True),
                          merge_mode='sum', name='CuDNN_bi_lstm')(x)
    else:
        x = Bidirectional(LSTM(units, activation='relu', kernel_initializer=kernel_init_rnn, dropout=dropout,
                               bias_initializer=bias_init_rnn, unit_forget_bias=True, return_sequences=True),
                          merge_mode='sum', name='bi_lstm')(x)

    # 1 fully connected relu layer
    x = TimeDistributed(Dense(units=units, kernel_initializer=kernel_init_dense, bias_initializer=bias_init_dense,
                              activation='relu'), name='fc_4')(x)
    x = TimeDistributed(Dropout(dropout), name='dropout_4')(x)

    # Output layer with softmax
    y_pred = TimeDistributed(Dense(units=output_dim, kernel_initializer=kernel_init_dense,
                                   bias_initializer=bias_init_dense, activation='softmax'), name='softmax')(x)

    # ---- CTC ----
    # y_input layers (transcription data) for CTC loss
    labels = Input(name='the_labels', shape=[None], dtype=dtype)  # transcription data (batch_size * y_seq_size)
    input_length = Input(name='input_length', shape=[1], dtype=dtype)  # unpadded len of all x_sequences in batch
    label_length = Input(name='label_length', shape=[1], dtype=dtype)  # unpadded len of all y_sequences in batch

    # Lambda layer with ctc_loss function due to Keras not supporting CTC layers
    loss_out = Lambda(function=ctc_lambda_func, name='ctc', output_shape=(1,))\
                     ([y_pred, labels, input_length, label_length])

    network_model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

    return network_model


# From Keras example https://github.com/keras-team/keras/blob/master/examples/image_ocr.py#L457
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    # print "y_pred_shape: ", y_pred.shape
    y_pred = y_pred[:, 2:, :]
    # print "y_pred_shape: ", y_pred.shape
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


# Returns clipped relu, clip value set to 20 (value from Baidu Deep speech 1)
def clipped_relu(value):
    return K.relu(value, max_value=20)
