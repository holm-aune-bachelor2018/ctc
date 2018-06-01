Speech recognition with RNN and CTC
======
Table of Contents
------
 * [Project structure](#project)
 * [Installation](#installation)
 * [Running](#running)
 * [Parameters](#params)
 * [Licence](#licence)

<a name="project"/>

## Project
Work in progress

<br>

<a name="installation"/>

## Installation
This project uses TensorFlow (version) and Keras (version).

This installation guide is for macOS and Ubuntu. 
TensorFlow also supports Windows but we have not tested this project on Windows.
1. Install Python  
This project uses Python 2.7  
Download and install from [Python download]

2. Install TensorFlow  
Install TensorFlow for Python 2.7 in a virtual environment following the [TensorFlow installation]  
If possible, GPU installation is recommended as it speeds up training significantly.

3. Install requirements  
Fork the project and download, or simply clone it, and enter the downloaded directory:
   ```
   $ git clone https://github.com/holm-aune-bachelor2018/ctc.git
   $ cd ctc
   ```
   Ensure that the python environment where you installed TensorFlow is active and install the             requirements:
   ```ubuntu
   $ source /home/<user>/tensorflow/bin/activate
   (tensorflow) $ pip install -r requirements.txt
   ```


**NOTE - multi GPU training**  
As per Keras (version), there is a bug when trying to save the model during training when using the multi_gpu_model().  
Please refer to this [Multi-GPU Model Keras guide] regarding how to save and load a multi-GPU model, including a work-around for the bug.

<br>

<a name="running"/>

## Running

**Download LibriSpeech** 

Ensure that the TensorFlow environment is active and that you are in the root project directory (ctc/)
This will download 55 GB of speech data into the data_dir directory

```
(tensorflow) $ import_librispeech.py data_dir 

```
<br> 

**Running training** <br>
If using TensorFlow with CPU or 1 GPU, to run the training with default parameters, simply do:
``` 
(tensorflow) $ train.py
```
This sets up training with the default BRNN model, using a small amount of data for testing. <br> 

**Example BRNN** <br>
Setting up a BRNN network, with 512 units, training on batch_size=64, epoch_len=256 <br>
That is, 64x256=16384 files or ~25 hours of data on the train-clean-360 dataset <br>
Train for epochs = 50 <br>
Save the .csv log file as "logs/brnn_25hours" <br>
Save the model every 10 epochs at "models/brnn_25hours.h5" <br>

```
(tensorflow) $ train.py --units=512 --batch_size=64 --epoch_len=256 --epochs=50 --model_type='brnn' --model_save='models/brnn_25hours.h5' --log_file='logs/brnn_25hours'  
```
<br> 

**Example loading** <br>
To continue training the same model for another 50 hours, use the model_load argument.
```
(tensorflow) $ train.py --model_load='models/brnn_25hours.h5' --units=512 --batch_size=64 --epoch_len=256 --epochs=50 --model_save='models/continued_brnn_25hours.h5' --log_file='logs/continued_brnn_25hours'  
```
<br> 

**Parallel GPU training** <br>
If running on multiple GPUs, enable multiGPU training:
```
(tensorflow) $ train.py --multi_GPU=2
```
Must be an even number of GPUs. <br> 

**Example CuDNNLSTM** <br>
ONLY WORKS WITH GPU <br>
With the GPU TensorFlow back you may wish to try the CuDNN optimised LSTM

```
(tensorflow) $ train.py --model_type=blstm --cudnn=True --units=512 --batch_size=64 --epoch_len=256 --epochs=50 --model_save='models/blstm_25hours.h5' --log_file='logs/blstm_25hours'
```

<br>

<a name="params"/>

## Parameters
Parameters for train.py

**Training params** <br>
```
--batch_size: Number of files in one batch. Default=32
--epoch_len: Number of batches per epoch. 0 trains on full dataset. Default=32
--epochs: Number of epochs to train. Default=10
--lr: Learning rate. Default=0.0001
--log_file: Path to log stats to .csv file. Default='logs'
```

**Multi GPU or single GPU / CPU training** <br>
```
--num_gpu: No. of gpu for training. (0,1) sets up training for one GPU or for CPU.
           MultiGPU training must be an even number larger than 1. Default=1
```

**Preprocessing params**<br>
```
--feature_type: What features to extract: mfcc, spectrogram. Default='mfcc'
--mfccs: Number of mfcc features per frame to extract. Default=26
--mels: Number of mels to use in feature extraction. Default=40
```

**Model params**<br>
```
--model_type: What model to train: brnn, blstm, deep_rnn, deep_lstm, cnn_blstm. Default='brnn'
--units: Number of hidden nodes. Default=256
--dropout: Set dropout value (0-1). Default=0.2
```

**Saving and loading model params**<br>
```
--model_save: Path, where to save model.
--checkpoint: No. of epochs before save during training. Default=10
--model_load: Path of existing model to load. If empty creates new model.
--load_multi: Load multi gpu model saved during parallel GPU training. Default=False
```

**Additional training settings**<br>
```
--save_best_val: Save additional version of model if val_loss improves. Defalt=False
--shuffle_indexes: If True, shuffle batches after each epoch. Default=False
--reduce_lr: Reduce the learning rate if model stops improving val_loss. Default=False
--early_stopping: Stop the training early if val_loss stops improving. Default=False
```

<a name="licence"/>
<br>

## Licence
This file is part of Speech recognition with CTC in Keras.

The project is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

The project is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this project. If not, see <http://www.gnu.org/licenses/>.
   


[Python download]: https://www.python.org/downloads/
[TensorFlow installation]: https://www.tensorflow.org/install/
[Multi-GPU Model Keras guide]: https://blog.datawow.io/multi-gpu-model-keras-ef463bf965d9
