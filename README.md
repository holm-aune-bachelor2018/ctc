Speech recognition with RNN and CTC 
======
[![Build Status](https://travis-ci.org/holm-aune-bachelor2018/ctc.svg?branch=master)](https://travis-ci.org/holm-aune-bachelor2018/ctc)

Table of Contents
------
 * [Project description](#project-description)
 * [Installation](#installation)
 * [Running](#running)
 * [Usage](#usage)
 * [Architecure overview](#architecture-overview)
 * [Licence](#licence)

## Project description
This repository is a part of TDAT3001 Bachelor Thesis in Computer Engineering at NTNU, project number 61:

"End-to-end speech recognition with recurrent neural networks and connectionist temporal classification" by Anita Kristine Aune and Marit Sundet-Holm, 2018.

The purpose of this project was to test different neural networks' performance on speech recognition, using recurrent neural networks (RNN) and connectionist temporal classification (CTC).

During this project we have tested various nettwork models for speech recognition. The resulting logs and saved models can be found at [train-logs].

## Installation
This project uses Python 2.7, TensorFlow version 1.6.1 and Keras version 2.1.5.

This installation guide is for macOS and Ubuntu. 
TensorFlow also supports Windows, but this project is not tested on Windows.
1. Install Python   
Download and install Python 2.7 from [Python download]

2. Install TensorFlow  
Install TensorFlow for Python 2.7 in a virtual environment following the [TensorFlow installation]  
If possible, GPU installation is recommended as it speeds up training significantly.

3. Install requirements  
Fork and download or clone the project, and enter the downloaded directory:
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
As per Keras version 2.1.5, there is a bug when trying to save the model during training when using the multi_gpu_model().  
Please refer to this [Multi-GPU Model Keras guide] regarding how to save and load a multi-GPU model, including a work-around for the bug.

## Running

**Download LibriSpeech** 

Ensure that the TensorFlow environment is active and that you are in the root project directory (ctc/)
This will download 55 GB of speech data into the data_dir directory

```
(tensorflow) $ import_librispeech.py data_dir 

```

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

**Example loading** <br>

To continue training the same model for another 50 epochs, use the model_load argument:
```
(tensorflow) $ train.py --model_load='models/brnn_25hours.h5' --units=512 --batch_size=64 --epoch_len=256 --epochs=50 --model_save='models/continued_brnn_25hours.h5' --log_file='logs/continued_brnn_25hours'  
```

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
(tensorflow) $ train.py --model_type=blstm --cudnn --units=512 --batch_size=64 --epoch_len=256 --epochs=50 --model_save='models/blstm_25hours.h5' --log_file='logs/blstm_25hours'
```

## Usage
```train.py``` is used to train models.
```predict.py``` is used to load already trained models, and produces predicions.

Parameters for ```train.py```:

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
--layers: Number of recurrent or deep layers. Default=1
--cudnn: Include to use cudnn optimized LSTM.
```

**Saving and loading model params**<br>
```
--model_save: Path, where to save model.
--checkpoint: No. of epochs before save during training. Default=10
--model_load: Path of existing model to load. If empty creates new model.
--load_multi: Include to load multi gpu model (saved during parallel GPU training).
```

**Additional training settings**<br>
```
--save_best_val: Include to save additional version of model if val_loss improves.
--shuffle_indexes: Include to shuffle batches after each epoch. 
--reduce_lr: Include to reduce the learning rate if model stops improving val_loss.
--early_stopping: Include to stop the training early if val_loss stops improving.
```

## Architecture overview
![alt text](https://github.com/holm-aune-bachelor2018/ctc/blob/master/images/architecture_overview.png)

Shows the overall structure of the project.  
  * <b>train.py</b> sets up network training.

  * <b>models.py</b> sets up the model build.
   
  * <b>data.py</b> generates a DataFrame containing filename (path to audio files), filesize and transcripts.
   
  * <b>DataGenerator.py</b> supplies the fit_generator() in train.py with batches of data during training.  
   
  * <b>LossCallback.py</b> is used by the fit_generator() in train to calculate WER and save model and logs during training.  
   
  * The remaning is varying utilities.  
   
   
   
Additionally, predict.py loads a trained model and creates prediction samples. It can also calculate WER.

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
[train-logs]: https://github.com/holm-aune-bachelor2018/train-logs
