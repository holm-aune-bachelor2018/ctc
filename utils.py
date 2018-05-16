from itertools import groupby

import numpy as np

from text import wers, int_to_text_sequence


def predict_batch(data_gen, test_func, batch_index):
    input_data, _ = data_gen.__getitem__(batch_index)

    x_data = input_data.get("the_input")
    y_data = input_data.get("the_labels")

    res = max_decode(test_func, x_data)
    predictions = []

    for i in range(y_data.shape[0]):
        original = "".join(int_to_text_sequence(y_data[i]))
        predicted = "".join(int_to_text_sequence(res[i]))
        predictions.append([original,predicted])

    return predictions


def calc_wer(test_func, data_gen):
    out_true = []
    out_pred = []
    for batch in xrange(0, data_gen.__len__(), data_gen.batch_size):
        input_data, _ = data_gen.__getitem__(batch)
        x_data = input_data.get("the_input")
        y_data = input_data.get("the_labels")

        for i in y_data:
            out_true.append("".join(int_to_text_sequence(i)))

        decoded = max_decode(test_func, x_data)
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
