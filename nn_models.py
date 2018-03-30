from keras.models import Model
from keras.layers import Dense, SimpleRNN, Bidirectional, Masking, TimeDistributed, Lambda, Input
import tensorflow as tf

# From Baidu Deep speech 1
def dnn_brnn(units, input_dim=12, output_dim=28):
    """
    :param units: units
    :param input_dim: input_dim(mfcc_features)
    :return: dnn_brnn model
    """
    # Model contains:
    # 1 layer of masking
    # 1 layer of batch normalization  ** Not yet implemented **
    # 3 layers of fully connected ReLu (DNN)
    # 1 layer of BRNN
    # 1 layer of ReLu
    # 1 layer of softmax

    # TODO: implement clipped ReLu? Dropout?

    x_data = Input(name='x_data',shape=(None, input_dim))
    y_true = Input(name='y_true', shape=[None])
    input_length = Input(name='y_pred_len', shape=[1])
    label_length = Input(name='y_true_len', shape=[1])

    # Masking layer
    x = Masking(mask_value=0.)(x_data)

    #TODO: batch norm layer

    # 3 fully connected layers DNN ReLu
    x = TimeDistributed(Dense(units=units, name='fc1', activation='relu'))(x)
    x = TimeDistributed(Dense(units=units, name='fc2', activation='relu'))(x)
    x = TimeDistributed(Dense(units=units, name='fc3', activation='relu'))(x)

    # Bidirectional RNN (with ReLu ?)
    x = Bidirectional(SimpleRNN(units, name='bi_rnn1',activation='relu', return_sequences=True))(x)

    # 1 fully connected relu layer + softmax
    x = TimeDistributed(Dense(units=units, name='fc4',activation='relu'))(x)

    # outout layer
    y_pred = TimeDistributed(Dense(units=output_dim, name='softmax',activation='softmax'))(x)

    ###### CTC ####

    # Lambda layer with ctc_loss function due to Keras not supporting CTC layers
    output = Lambda(ctc_loss, name='ctc', output_shape=(1,))([y_true, y_pred, input_length, label_length])

    model = Model(inputs=[x_data, y_true, input_length, label_length], outputs=output)

    # model.summary() #prints summary
    return model

# Calculates ctc loss via TensorFlow ctc_batch_cost
def ctc_loss(args):
    y_true = args[0]
    y_pred = args[1]
    input_length = args[2]
    label_length = args[3]

    return tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)