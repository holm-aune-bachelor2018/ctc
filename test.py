import preprocessing

path = "sample_data/dev-clean/LibriSpeech/dev-clean/84/121123/84-121123-0000.flac"
frame_length = 400
hop_length = 160

mfcc = preprocessing.mfcc(path, frame_length, hop_length)

print mfcc

