import preprocessing
import lstm
import numpy as np
import pandas as pd
import data

#Preprocessing
path = "sample_data/wav_sample/sample_librivox-dev-clean.csv"
#path = "sample_data/wav_sample/sample_librivox-dev-clean.csv"
frequency = 16
frame_length = 20*frequency
hop_length = 10*frequency


dataprop, df_final = data.combine_all_wavs_and_trans_from_csvs(path)

print preprocessing.generate_batch(df_final, frame_length, hop_length)



df_final.to_csv('data.csv', index=False)


#Model
units = 512                         #numb of hidden nodes
mfcc_features = 12                  #input_dim
input_shape=(None, mfcc_features)   #None to be able to process batches of any size



#lstm.simpleLSTM(units, input_shape=input_shape, X=dataX, Y=dataY)

