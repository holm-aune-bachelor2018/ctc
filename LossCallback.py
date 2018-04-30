from keras.callbacks import Callback
from keras import Model
from itertools import groupby
import numpy as np
from text import wers, int_to_text_sequence


class LossCallback(Callback):
    """


    Args:
        test_func ( ):
        validation_gen ( ):

    """
    def __init__(self, test_func, validation_gen, model, checkpoint, path_to_save):
        self.test_func = test_func
        self.validation_gen = validation_gen
        self.model = model
        self.checkpoint = checkpoint
        self.path_to_save = path_to_save

    def on_epoch_end(self, epoch, logs={}):
        wers = self.calc_wer()
        print " - average WER: ", wers[1]

        value_list = []
        value_list.append(logs.get('loss'))
        value_list.append(logs.get('val_loss'))
        value_list.append(wers[1])

        self.values.append(value_list)

        if ((epoch+1) % self.checkpoint) == 0 and self.path_to_save:
            model_to_save = Model(self.model.inputs, self.model.outputs)
            model_to_save.save(self.path_to_save)

    def on_train_begin(self, logs={}):
        self.values = []

    def on_train_end(self, logs={}):
        print "\n - Training ended, prediction samples -"
        batch = 6
        input, output = self.validation_gen.__getitem__(batch)

        x_data = input.get("the_input")
        y_data = input.get("the_labels")

        res = max_decode(self.test_func, x_data)
        for i in range(y_data.shape[0]):
            print "Original: ","".join(int_to_text_sequence(y_data[i]))
            print "Predicted: ","".join(int_to_text_sequence(res[i])), "\n"

    def calc_wer(self):
        out_true=[]
        out_pred=[]
        for batch in xrange(0, self.validation_gen.epoch_length, self.validation_gen.batch_size):
            input, output = self.validation_gen.__getitem__(batch)
            x_data = input.get("the_input")
            y_data = input.get("the_labels")

            for i in y_data:
                out_true.append("".join(int_to_text_sequence(i)))

            decoded = max_decode(self.test_func, x_data)
            for i in decoded:
                out_pred.append("".join(int_to_text_sequence(i)))

        out = wers(out_true, out_pred)

        return out


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
