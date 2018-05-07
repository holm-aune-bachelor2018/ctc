from keras.callbacks import Callback
from keras import Model
from itertools import groupby
import numpy as np
from text import wers, int_to_text_sequence
from datetime import datetime
import pandas


class LossCallback(Callback):

    def __init__(self, test_func, validation_gen, test_gen, model, checkpoint, path_to_save, log_file_path):
        self.test_func = test_func
        self.validation_gen = validation_gen
        self.test_gen = test_gen
        self.model = model
        self.checkpoint = checkpoint
        self.path_to_save = path_to_save
        self.log_file_path = log_file_path
        self.timestamp = datetime.now().strftime('%m-%d_%H%M') + ".csv"

    def on_epoch_end(self, epoch, logs={}):
        wer = self.calc_wer(self.validation_gen)
        print " - average WER: ", wer[1]

        value_list = []
        value_list.append(logs.get('loss'))
        value_list.append(logs.get('val_loss'))
        value_list.append(wer[1])

        self.values.append(value_list)

        if ((epoch+1) % self.checkpoint) == 0:
            if self.path_to_save:
                model_to_save = Model(self.model.inputs, self.model.outputs)
                model_to_save.save(self.path_to_save)
            self.save_log()

    def on_train_begin(self, logs={}):
        self.values = []

    def on_train_end(self, logs={}):
        test_wer = self.calc_wer(self.test_gen)
        print "\n - Training ended, test wer: ", test_wer[1]
        print "\nPrediction samples -"
        batch = 6
        input, output = self.validation_gen.__getitem__(batch)

        x_data = input.get("the_input")
        y_data = input.get("the_labels")

        res = max_decode(self.test_func, x_data)
        for i in range(y_data.shape[0]):
            print "Original: ","".join(int_to_text_sequence(y_data[i]))
            print "Predicted: ","".join(int_to_text_sequence(res[i])), "\n"

        self.save_log()

    def calc_wer(self, data_gen):
        out_true=[]
        out_pred=[]
        for batch in xrange(0, data_gen.epoch_length, data_gen.batch_size):
            input, output = data_gen.__getitem__(batch)
            x_data = input.get("the_input")
            y_data = input.get("the_labels")

            for i in y_data:
                out_true.append("".join(int_to_text_sequence(i)))

            decoded = max_decode(self.test_func, x_data)
            for i in decoded:
                out_pred.append("".join(int_to_text_sequence(i)))

        out = wers(out_true, out_pred)

        return out

    def save_log(self):
        stats = pandas.DataFrame(data=self.values, columns=['loss', 'val_loss', 'wer'])
        stats.to_csv(self.log_file_path + "_" + self.timestamp)
        print "Log file saved: ", self.log_file_path + "_" + self.timestamp


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
