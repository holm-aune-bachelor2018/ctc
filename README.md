Speech recognition with RNN and CTC
======
Table of Contents
------
 * [Project structure](#project)
 * [Installation](#installation)
 * [Running](#running)
 * [Licence](#licence)

<a name="project"/>

## Project
Work in progress

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
   ```Ubuntu
   $ source /home/<user>/tensorflow/bin/activate
   (tensorflow) $ pip install -r requirements.txt
   ```


**NOTE - multi GPU training**  
As per Keras (version), there is a bug when trying to save the model during training when using the multi_gpu_model().  
Please refer to this [Multi-GPU Model Keras guide] regarding how to save and load a multi-GPU model, including a work-around for the bug.

<a name="running"/>

## Running
Work in progress

<a name="licence"/>

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
