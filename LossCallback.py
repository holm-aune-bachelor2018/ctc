from keras.callbacks import Callback
# import itertools
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
            batch = 0
            input, output = self.validation_gen.__getitem__(batch)

            x_data = input.get("the_input")
            input_length = input.get("input_length")

            # res = epoch_stats(self.test_func, x_data)
            # print "Res epoch stats: ", res
            res = decode_batch(self.test_func, x_data, input_length)
            # print "Res of decode batch: ", res


# K.ctc_decode?
# For a real OCR application, this should be beam search with a dictionary
# and language model.  For this example, best path is sufficient.
def decode_batch(test_func, x_data, input_length=0):
    y_pred = test_func([x_data])[0]
    print "y_pred shape: ", y_pred.shape
    print "y_pred 0 shape :", y_pred[0].shape
    array = np.arange(y_pred[0].shape[0])
    print "Array shape: ", array.shape
    res0 = np.argmax(y_pred[4], axis=1)
    res1 = np.argmax(y_pred[6], axis=1)

    print "\nBatch0 maximum", res0
    print "\nBatch1 maximum", res1

    # K.ctc_decode(y_pred, input_length, greedy=True, beam_width=100, top_paths=1)
    # tuple = K.ctc_decode(y_pred, input_length)
    # print "tuple [0][0] ", tuple[0][0]
    # print "tuple [1][0] ", tuple[1][0]


    # res = []
    # for j in range(y_pred.shape[0]):
    #    out_best = list(np.argmax(y_pred[j, 2:], 1))
    #    out_best = [k for k, g in itertools.groupby(out_best)]
    #    print "Pred in int: ", out_best
    #    outstr = int_to_text_sequence(out_best)
    #    res.append(outstr)
    return res0


def epoch_stats(test_func, input):
    y_pred = test_func([input])[0]
    print "y prEd: ", y_pred.shape
    y_orig = "abc"

    res = wer(y_orig, y_pred)
    return res
