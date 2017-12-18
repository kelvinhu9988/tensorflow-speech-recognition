#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 19:34:07 2017

@author: william
"""

#   EDA and preprocessing
import os
import numpy as np 
import pandas as pd 
import pickle
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from pathlib import Path
from scipy import signal
from scipy.io import wavfile
from sklearn.preprocessing import LabelEncoder

np.random.seed(1104) #  for consistency

#---------------------------------------------------#
#--------------   Data Labeling   ------------------#
#---------------------------------------------------#
train_audio_path = '../input/train/audio'

train_labels = os.listdir(train_audio_path)
train_labels.remove('_background_noise_')

labels_to_keep = ['yes', 'no', 'up', 'down', 'left',
                  'right', 'on', 'off', 'stop', 'go', 'silence'] 

train_file_labels = dict()
for label in train_labels:
    files = os.listdir(train_audio_path + '/' + label)
    for f in files:
        train_file_labels[label + '/' + f] = label

train = pd.DataFrame.from_dict(train_file_labels, orient='index')
train = train.reset_index(drop=False)
train = train.rename(columns={'index': 'file', 0: 'folder'})
train = train[['folder', 'file']]
train = train.sort_values('file')
train = train.reset_index(drop=True)

train['file'] = train['file'].apply(lambda x: x.split('/')[1])
train['label'] = train['folder'].apply(lambda x: x if x in labels_to_keep else 'unknown')

#---------------------------------------------------#
#---------------   Spectrogram   -------------------#
#---------------------------------------------------#

#   For each of the file, convert them into spectrogram
#   and they would act as the training set

train_audio_path = '../input/train/audio/'

spectrogram_container = []

def log_specgram(audio, sample_rate, window_size=20,
                 step_size=10, eps=1e-10):
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    freqs, times, spec = signal.spectrogram(audio,
                                    fs=sample_rate,
                                    window='hann',
                                    nperseg=nperseg,
                                    noverlap=noverlap,
                                    detrend=False)
    return freqs, times, np.log(spec.T.astype(np.float32) + eps)

def to_spectrogram(filename):
    sample_rate, samples = wavfile.read(filename)
    freqs, times, spectrogram = log_specgram(samples, sample_rate)
    spectrogram_container.append(spectrogram)
    
#   Transform each audio into spectrograms and store them in spectrogram_container
#   with the same index as train     
train.apply(lambda x: to_spectrogram(train_audio_path + x[0] + '/' + x[1]), axis=1)

#   Store the data in pickle 
#   PS: Errno 22, the file is too large to be stored in a pickle file
#with open("../input/train/spectrogram.pickle", 'wb') as handle:
#    pickle.dump(spectrogram_container, handle)



#---------------------------------------------------#
#--------   Training and Testing Split   -----------#
#---------------------------------------------------#


#---------------------------------------------------#
#------------   CNN Model Training   ---------------#
#---------------------------------------------------#

#   One-hot encoding
y = train['label']
encoder = LabelEncoder()
encoder.fit(y)
label = np_utils.to_categorical(encoder.transform(y))

#   Define model

## 3. Import libraries and modules
#import numpy as np
#np.random.seed(123)  # for reproducibility
# 
#from keras.models import Sequential
#from keras.layers import Dense, Dropout, Activation, Flatten
#from keras.layers import Convolution2D, MaxPooling2D
#from keras.utils import np_utils
#from keras.datasets import mnist
# 
## 4. Load pre-shuffled MNIST data into train and test sets
#(X_train, y_train), (X_test, y_test) = mnist.load_data()
# 
## 5. Preprocess input data
#X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
#X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
#X_train = X_train.astype('float32')
#X_test = X_test.astype('float32')
#X_train /= 255
#X_test /= 255
# 
## 6. Preprocess class labels
#Y_train = np_utils.to_categorical(y_train, 10)
#Y_test = np_utils.to_categorical(y_test, 10)
# 
## 7. Define model architecture
#model = Sequential()
# 
#model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(1,28,28)))
#model.add(Convolution2D(32, 3, 3, activation='relu'))
#model.add(MaxPooling2D(pool_size=(2,2)))
#model.add(Dropout(0.25))
# 
#model.add(Flatten())
#model.add(Dense(128, activation='relu'))
#model.add(Dropout(0.5))
#model.add(Dense(10, activation='softmax'))
# 
## 8. Compile model
#model.compile(loss='categorical_crossentropy',
#              optimizer='adam',
#              metrics=['accuracy'])
# 
## 9. Fit model on training data
#model.fit(X_train, Y_train, 
#          batch_size=32, nb_epoch=10, verbose=1)
# 
## 10. Evaluate model on test data
#score = model.evaluate(X_test, Y_test, verbose=0)
















