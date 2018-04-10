import keras
import itertools
import numpy as np
import keras.backend as K
from data import int_to_text_sequence
from data import text_to_int_sequence

"""
class VizCallback(keras.callbacks.Callback):

def __init__(self, run_name, test_func, text_img_gen, num_display_words=6):
    self.test_func = test_func
    self.output_dir = os.path.join(
        OUTPUT_DIR, run_name)
    self.text_img_gen = text_img_gen
    self.num_display_words = num_display_words
    if not os.path.exists(self.output_dir):
    os.makedirs(self.output_dir)
"""


class LossCallback(keras.callbacks.Callback):

    def __init__(self, test_func, validation_gen):
        self.test_func = test_func
        self.validation_gen = validation_gen

#    def on_train_begin(self, logs={}):

    def on_epoch_end(self, epoch, logs={}):
        if epoch == 1:
            input, output = self.validation_gen.__getitem__(0)
            res = decode_batch(self.test_func, input.get("the_input"))
            print "Res of decode batch: ", res


# K.ctc_decode?
# For a real OCR application, this should be beam search with a dictionary
# and language model.  For this example, best path is sufficient.
def decode_batch(test_func, input, input_length=0):
    y_pred = test_func([input])[0]
    ret = []
    # K.ctc_decode(y_pred, input_length, greedy=True, beam_width=100, top_paths=1)
    # ret = K.ctc_decode(y_pred, input_length)
    for j in range(y_pred.shape[0]):
        out_best = list(np.argmax(y_pred[j, 2:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        print "Pred in int: ", out_best
        outstr = int_to_text_sequence(out_best)
        ret.append(outstr)
    return ret