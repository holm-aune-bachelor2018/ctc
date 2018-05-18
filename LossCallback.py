from datetime import datetime

import pandas
from keras import Model
from keras.callbacks import Callback

from utils import calc_wer, predict_batch


class LossCallback(Callback):
    def __init__(self, test_func, validation_gen, test_gen, model, checkpoint, path_to_save, log_file_path):
        self.test_func = test_func
        self.validation_gen = validation_gen
        self.test_gen = test_gen
        self.model = model
        self.checkpoint = checkpoint
        self.path_to_save = path_to_save
        self.log_file_path = log_file_path
        self.values = []
        self.timestamp = datetime.now().strftime('%m-%d_%H%M') + ".csv"

    def on_epoch_end(self, epoch, logs={}):
        wer = calc_wer(self.test_func, self.validation_gen)
        print " - average WER: ", wer[1], "\n"

        self.values.append([logs.get('loss'), logs.get('val_loss'), wer[1]])

        if ((epoch+1) % self.checkpoint) == 0:
            if self.path_to_save:
                model_to_save = Model(self.model.inputs, self.model.outputs)
                model_to_save.save(self.path_to_save)
            self.save_log()

    def on_train_end(self, logs={}):
        try:
            test_wer = calc_wer(self.test_func, self.test_gen)
            print "\n - Training ended, test wer: ", test_wer[1], " -"
        except (Exception, ArithmeticError) as e:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(e).__name__, e.args)
            print message

        # Print a sample of predictions, for visualisation
        print "\nPrediction samples:\n"
        predictions = predict_batch(self.validation_gen, self.test_func, 6)

        for i in predictions:
            print "Original: ", i[0]
            print "Predicted: ", i[1], "\n"

        self.save_log()

    def save_log(self):
        stats = pandas.DataFrame(data=self.values, columns=['loss', 'val_loss', 'wer'])
        stats.to_csv(self.log_file_path + "_" + self.timestamp)
        print "Log file saved: ", self.log_file_path + "_" + self.timestamp


