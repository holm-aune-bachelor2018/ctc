import numpy as np
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM, Bidirectional, Masking, BatchNormalization, TimeDistributed
from keras.utils import np_utils

np.random.seed(7)
loss = 'categorical_crossentropy'
optimizer = 'adam'
metrics = ['accuracy']


# from baidu deep speech 1
def dnn_brnn(units, input_shape, X, y):
    """
    :param units:
    :param input_shape: (batch_size(number of seq), time_steps(sequence length), input_dim(mfcc_features))
    :param X:
    :param Y:
    :return:
    """
    # 1 layer of masking
    # 1 layer of batch normalization?
    # 3 layers of DNN ReLu
    # 1 layer of BRNN
    # 1 layer of DNN ReLu
    # 1 layer of softmax

    # Model compiling and fit
    model = Sequential()

    model.add(Masking(mask_value=0., input_shape=input_shape))
    #model.add(BatchNormalization())

    # TODO: implement clipped ReLu? Dropout?
    # TODO: model.add? or x= ...
    model.add(TimeDistributed(Dense(units=units, activation='relu'), input_shape=input_shape))
    model.add(TimeDistributed(Dense(units=units, activation='relu'), input_shape=input_shape))
    model.add(TimeDistributed(Dense(units=units, activation='relu'), input_shape=input_shape))

    model.add(Bidirectional(SimpleRNN(units, activation='relu', return_sequences=True)))

    model.add(TimeDistributed(Dense(units=units, activation='relu'), input_shape=input_shape))

    model.add(Dense(y, activation='softmax'))

    # Add CTC


    return model


'''
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    model.fit(X,y, epochs=100, batch_size=1, verbose=0)

    #Results
    scores = model.evaluate(X, y, verbose=0)
    return scores
'''
