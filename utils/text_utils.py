from utils.char_map import char_map
from utils.char_map import index_map

# The following code is adapted from: github.com/baidu-research/ba-dls-deepspeech
# Which is under the Apache License:
#    Copyright 2015-2016 Baidu USA LLC.  All rights reserved.

#    Apache License
#    Version 2.0, January 2004
#    http://www.apache.org/licenses/


def text_to_int_sequence(text):
    """ Use a character map and convert text to an integer sequence """
    int_sequence = []
    for c in text:
        if c == ' ':
            ch = char_map['<SPACE>']
        else:
            ch = char_map[c]
        int_sequence.append(ch)
    return int_sequence


def int_to_text_sequence(seq):
    """ Use a index map and convert int to a text sequence """
    text_sequence = []
    for c in seq:
        if c == 28:  # ctc/pad char
            ch = ''
        else:
            ch = index_map[c]
        text_sequence.append(ch)
    return text_sequence
