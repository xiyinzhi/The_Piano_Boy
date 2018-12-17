# The_Piano_Boy
A deep neural network that can compose some piano music by learning from piano music datasets in MIDI format using a 3-layer LSTM model.

## Requirements
Python 3.x
### Installing the following packages using pip3:
Music21

Keras

Tensorflow

h5py

numpy

## Models
We provide 3 models of 3-layer LSTM with different improvements.

• one initial model, its input and output contains rest notes.

We employed the follwing two different models with 88 keys representation:

• 88notes_sigmoid.ipynb, it predicts multiple possible keys at one beat, it use sigmoid as activation function of output layer and binary crossentropy as loss function, we also use a threshold 0.5 to determine if a key is pressed or not.

• 88notes_softmax.ipynb, it only predict the key with the most probability, it use softmax as activation function of output layer and categorical crossentropy as loss function.

Generally speaking, 88 keys model 1 is the best optimized one. You are free to try all models for comparison.

## How to run

You can just run our .ipynb files in Jupyter Notebook. 

After training, replace the weights with the best one in the predict part and generate a piece of amazing new music!


Now, start your journey!





