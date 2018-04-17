from keras.callbacks import Callback
from itertools import groupby
import numpy as np
from text import wer, int_to_text_sequence, text_to_int_sequence
# from keras import backend as K


class LossCallback(Callback):

    def __init__(self, test_func, validation_gen):
        self.test_func = test_func
        self.validation_gen = validation_gen

#    def on_train_begin(self, logs={}):

    def on_epoch_end(self, epoch, logs={}):
        if (epoch%5 == 0):
            print "\n Epoch: ", epoch

            batch = 2
            input, output = self.validation_gen.__getitem__(batch)

            x_data = input.get("the_input")
            y_data = input.get("the_labels")
            print "\n##########"
            print "Y TRUE: "
            for i in y_data:
                print "".join(int_to_text_sequence(i))

            # res = epoch_stats(self.test_func, x_data)
            # print "Res epoch stats: ", res
            res = max_decode(self.test_func, x_data)
            print "\nRES of max decode: ", res, "\nIn text: "
            for i in res:
                print "".join(int_to_text_sequence(i))
            print "##########\n"


def max_decode(test_func, x_data):
    y_pred = test_func([x_data])[0]

    decoded = []
    for i in range(0,y_pred.shape[0]):

        decoded_batch = []
        for j in range(0,y_pred.shape[1]):
            decoded_batch.append(np.argmax(y_pred[i][j]))

        temp = [k for k, g in groupby(decoded_batch)]
        temp[:] = [x for x in temp if x != [28]]
        decoded.append(temp)

    return decoded


def epoch_stats(test_func, input):
    y_pred = test_func([input])[0]
    print "y prEd: ", y_pred.shape
    y_orig = "abc"

    res = wer(y_orig, y_pred)
    return res
