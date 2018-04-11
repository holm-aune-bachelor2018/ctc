import keras
import itertools
import numpy as np
from text import wer, int_to_text_sequence, text_to_int_sequence
from keras import backend as K


class LossCallback(keras.callbacks.Callback):

    def __init__(self, test_func, validation_gen):
        self.test_func = test_func
        self.validation_gen = validation_gen

#    def on_train_begin(self, logs={}):

    def on_epoch_end(self, epoch, logs={}):
        input, output = self.validation_gen.__getitem__(0)

        res = epoch_stats(self.test_func, input.get("the_input"))
        print "Res epoch stats: ", res
        #res = decode_batch(self.test_func, input.get("the_input"))
        #print "Res of decode batch: ", res


# K.ctc_decode?
# For a real OCR application, this should be beam search with a dictionary
# and language model.  For this example, best path is sufficient.
def decode_batch(test_func, input, input_length=0):
    y_pred = test_func([input])[0]
    ret = []

    #K.ctc_decode(y_pred, input_length, greedy=True, beam_width=100, top_paths=1)

    ret = K.ctc_decode(y_pred, input_length)
    for j in range(y_pred.shape[0]):
        out_best = list(np.argmax(y_pred[j, 2:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        print "Pred in int: ", out_best
        outstr = int_to_text_sequence(out_best)
        ret.append(outstr)
    return ret


def epoch_stats(test_func, input):
    y_pred = test_func([input])[0]
    print "y prEd: ", y_pred.shape
    y_orig = "abc"

    res = wer(y_orig, y_pred)
    return res
