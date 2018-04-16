from keras.models import Model
from keras.layers import Dense, SimpleRNN, LSTM, Bidirectional, Masking, TimeDistributed, Lambda, Input, Dropout, BatchNormalization
from keras import backend as K


# Architecture from Baidu Deep speech 1
def dnn_brnn(units, input_dim=12, output_dim=29):
    """
    :param units: units
    :param input_dim: input_dim(mfcc_features)
    :param output_dim output dim of final layer of model, input to CTC layer
    :return: dnn_brnn model

    Model contains:
     1 layer of masking
     3 layers of fully connected clipped ReLu (DNN) with dropout 10 % between each layer
     1 layer of BRNN
     1 layer of ReLu
     1 layer of softmax
    """

    # x_input layer, dim: (batch_size * x_seq_size * mfcc_features)
    input_data = Input(name='the_input',shape=(None, input_dim))

    # Masking layer
    x = Masking(mask_value=0.)(input_data)

    # 3 fully connected layers DNN ReLu
    # Dropout rate 10 % at each FC layer

    x = TimeDistributed(Dropout(0.2), name='dropout_1')(x)
    x = TimeDistributed(Dense(units=units, name='fc1', kernel_initializer='random_normal', activation=clipped_relu), name='fc_1')(x)

    x = TimeDistributed(Dropout(0.2), name='dropout_2')(x)
    x = TimeDistributed(Dense(units=units, name='fc2', kernel_initializer='random_normal', activation=clipped_relu), name='fc_2')(x)

    x = TimeDistributed(Dropout(0.2), name='dropout_3')(x)
    x = TimeDistributed(Dense(units=units, name='fc3', kernel_initializer='random_normal', activation=clipped_relu), name='fc_3')(x)

    # TODO: mergemode? default = concat, kernel_initializer? bias_initializer?
    # Bidirectional RNN (with ReLu ?)
    # x = BatchNormalization()(x)
    x = Bidirectional(SimpleRNN(units, name='bi_rnn1', activation='relu', return_sequences=True),
                      merge_mode='sum', name='bi_rnn')(x)

    # 1 fully connected relu layer + softmax
    inner = TimeDistributed(Dense(units=units, name='fc4', kernel_initializer='random_normal', activation='relu'), name='fc_4')(x)

    # Output layer
    y_pred = TimeDistributed(Dense(units=output_dim, name='softmax', activation='softmax'), name='softmax')(inner)

    ###### CTC ####
    # y_input layers (transcription data) for CTC loss
    labels = Input(name='the_labels', shape=[None], dtype='float32')        # transcription data (batch_size * y_seq_size)
    input_length = Input(name='input_length', shape=[1], dtype='float32')   # unpadded len of all x_sequences in batch (batch_size * 1)
    label_length = Input(name='label_length', shape=[1], dtype='float32')   # unpadded len of all y_sequences in batch (batch_size * 1)

    # Model(inputs=input_data, outputs=y_pred).summary()

    # Lambda layer with ctc_loss function due to Keras not supporting CTC layers
    loss_out = Lambda(ctc_lambda_func, name='ctc', output_shape=(1,))([y_pred, labels, input_length, label_length])

    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

    return model


# From keras example https://github.com/keras-team/keras/blob/master/examples/image_ocr.py#L457
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    # print "y_pred_shape: ", y_pred.shape
    y_pred = y_pred[:, 2:, :]
    # print "y_pred_shape: ", y_pred.shape
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


"""
# Calculates ctc loss via TensorFlow ctc_batch_cost
def ctc_loss(args):
    y_true = args[0]
    y_pred = args[1]
    input_length = args[2]
    label_length = args[3]

    return K.ctc_batch_cost(y_true, y_pred, input_length, label_length)
"""

# Returns clipped relu, clip value set to 20 (value from Baidu Deep speech 1)
def clipped_relu(value):
    return K.relu(value, max_value=20)