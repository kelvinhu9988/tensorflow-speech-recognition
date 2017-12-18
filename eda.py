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
import datetime
from time import gmtime, strftime
from collections import Counter 
from keras import optimizers, losses, activations, models
from keras.layers import Convolution2D, Dense, Input, Flatten, Dropout, MaxPooling2D, BatchNormalization
from keras.utils import np_utils
from keras.models import Sequential
from scipy import signal
from scipy.io import wavfile
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

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

audio_dim = 16000

spectrogram_container = []
sample_container = []

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
    #   Pad the same to audio size(16000) so that the spectrogram dimension agree
    samples = np.array(list(samples) + [0 for x in range(audio_dim - len(samples))])
    freqs, times, spectrogram = log_specgram(samples, sample_rate)
    
    #sample_container.append(samples)
    spectrogram_container.append(spectrogram)
    
#   Transform each audio into spectrograms and store them in spectrogram_container
#   with the same index as train        
train.apply(lambda x: to_spectrogram(train_audio_path + x[0] + '/' + x[1]), axis=1)
spectrogram_container = np.array(spectrogram_container)
spectrogram_container = spectrogram_container.reshape(list(spectrogram_container.shape) + [1])

#   Store the data in pickle 
#   PS: Errno 22, the file is too large to be stored in a pickle file
#with open("../input/train/spectrogram.pickle", 'wb') as handle:
#    pickle.dump(spectrogram_container, handle)



#---------------------------------------------------#
#--------   Training and Testing Split   -----------#
#---------------------------------------------------#

#   the same person should be in the same set(train/test)


#---------------------------------------------------#
#------------   CNN Model Training   ---------------#
#---------------------------------------------------#

#   One-hot encoding
y = train['label']
encoder = LabelEncoder()
encoder.fit(y)
label = np_utils.to_categorical(encoder.transform(y))

#   Define model


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
# 7. Define model architecture
model = Sequential()
model.add(Convolution2D(8, kernel_size=2, activation='relu', input_shape=(99,161,1)))
model.add(Convolution2D(8, kernel_size=2, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Convolution2D(16, kernel_size=2, activation='relu'))
model.add(Convolution2D(16, kernel_size=2, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Convolution2D(32, kernel_size=2, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(11, activation='softmax'))
 
# 8. Compile model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

x_train, x_valid, y_train, y_valid = train_test_split(spectrogram_container, label, 
                                                      test_size=0.1, random_state=2017)

## 9. Fit model on training data
model.fit(x_train, y_train, 
          batch_size=32, epochs=2, verbose=1,
          validation_data=(x_valid,y_valid))

model.save('../model/' + strftime("%Y-%m-%d %H:%M:%S", gmtime()) + '.h5')
 





#---------------------------------------------------#
#-----------------   Testing   ---------------------#
#---------------------------------------------------#


#def test_data_generator(batch=16):
#    fpaths = glob(os.path.join(test_data_path, '*wav'))
#    i = 0
#    for path in fpaths:
#        if i == 0:
#            imgs = []
#            fnames = []
#        i += 1
#        rate, samples = wavfile.read(path)
#        samples = pad_audio(samples)
#        resampled = signal.resample(samples, int(new_sample_rate / rate * samples.shape[0]))
#        _, _, specgram = log_specgram(resampled, sample_rate=new_sample_rate)
#        imgs.append(specgram)
#        fnames.append(path.split('\\')[-1])
#        if i == batch:
#            i = 0
#            imgs = np.array(imgs)
#            imgs = imgs.reshape(tuple(list(imgs.shape) + [1]))
#            yield fnames, imgs
#    if i < batch:
#        imgs = np.array(imgs)
#        imgs = imgs.reshape(tuple(list(imgs.shape) + [1]))
#        yield fnames, imgs
#    raise StopIteration()















