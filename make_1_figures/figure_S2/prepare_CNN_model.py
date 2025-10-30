import h5py
import pickle
import scipy
import numpy as np
import seaborn as sns
# sns.set()
sns.set_style("whitegrid", {'axes.grid' : False})
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
from os.path import join as pjoin
import scipy.io as sio
from scipy import stats
import scipy.signal as sp
from scipy import signal
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from statannot import add_stat_annotation
from skimage import io
from skimage.feature import hog
from joblib import Parallel, delayed
import cv2
import glob
import time
from skimage.feature import register_translation
from scipy.ndimage import fourier_shift
from joblib import Parallel, delayed
from skimage.feature import hog
import sys
sys.path.insert(1, '/home/wx2203/Weihao/code_locker/common_functions')
from harness import *
from f2_utils import *
import matplotlib.patches as patches

from IPython.display import display, clear_output, HTML, Image
from matplotlib import animation, rc

from skimage.measure import regionprops

from matplotlib import cm as mpl_cm
from matplotlib import (colors,
                        lines,
                        transforms,
                        )
from matplotlib import patches

import time_in_each_roi 

from TimeseriesGenerator import TimeseriesGenerator
from sklearn.linear_model import MultiTaskLassoCV,LassoCV
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, LSTM, Input, Lambda, Bidirectional, Conv1D
from keras.models import Model, Sequential
from keras.regularizers import l1_l2
from keras.utils import np_utils
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

def prepare_model(loss_func,metric, timesteps, num_in_features, num_out_features):


    input_layer = Input(shape=(timesteps, num_in_features))
    conv1 = Conv1D(filters=1024,
               kernel_size=20,
               strides=1,
               activation='relu',kernel_regularizer=l1_l2(l1=0.0001, l2=0.0001),
               padding='same')(input_layer)
    lstm1 = Bidirectional(LSTM(100, return_sequences=True,
                               kernel_regularizer=l1_l2(l1=0.0001, l2=0.0001)))(conv1)
    dense1 = Dense(128,activation='tanh')(lstm1)
    dense2 = Dense(64,activation='tanh')(dense1)
    output_layer = Dense(num_out_features)(dense2)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss=loss_func, optimizer='adam',metrics=[metric])
    return model

def prepare_model_without_lstm(loss_func,metric, timesteps, num_in_features, num_out_features):


    input_layer = Input(shape=(timesteps, num_in_features))
    conv1 = Conv1D(filters=1024,
               kernel_size=20,
               strides=1,
               activation='relu',kernel_regularizer=l1_l2(l1=0.0001, l2=0.0001),
               padding='same')(input_layer)
    # lstm1 = Bidirectional(LSTM(100, return_sequences=True,
    #                            kernel_regularizer=l1_l2(l1=0.0001, l2=0.0001)))(conv1)
    dense1 = Dense(128,activation='tanh')(conv1)
    dense2 = Dense(64,activation='tanh')(dense1)
    output_layer = Dense(num_out_features)(dense2)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss=loss_func, optimizer='adam',metrics=[metric])
    return model

def prepare_model_linear_act(loss_func,metric, timesteps, num_in_features, num_out_features):


    input_layer = Input(shape=(timesteps, num_in_features))
    conv1 = Conv1D(filters=1024,
               kernel_size=20,
               strides=1,
               activation='linear',kernel_regularizer=l1_l2(l1=0.0001, l2=0.0001),
               padding='same')(input_layer)
    # lstm1 = Bidirectional(LSTM(100, return_sequences=True,activation='linear',
    #                            kernel_regularizer=l1_l2(l1=0.0001, l2=0.0001)))(conv1)
    dense1 = Dense(128,activation='linear')(conv1)
    dense2 = Dense(64,activation='linear')(dense1)
    output_layer = Dense(num_out_features)(dense2)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss=loss_func, optimizer='adam',metrics=[metric])
    return model

def fit_model(model, model_weights_filename, X_train, Y_train, X_validation,
              Y_validation, seed, num_epochs=40, patience=50, verbosity=0,
              batch_size=64):
    save = ModelCheckpoint(model_weights_filename, monitor='val_loss',
                           verbose=0, save_best_only=True, mode='auto')
    early = EarlyStopping(monitor='val_loss', min_delta=0, patience=patience,
                          verbose=0, mode='auto')
    callbacks_list = [save, early]
    # set_random_seed(seed)
    # np.random.seed(seed)
    history = model.fit(X_train, Y_train,
                        validation_data=(X_validation, Y_validation),
                        callbacks=callbacks_list,
                        epochs=num_epochs, batch_size=batch_size,
                        verbose=verbosity)
    return model, history


def load_model(model, model_weights_filename):
    model.load_weights(model_weights_filename)
    return model

def load_prepare_data(X_continued, y_continued,length=40,stride_length=5,num_regions=500):


    # slice the time series
    data_gen = TimeseriesGenerator(X_continued,y_continued,
                                   stride = stride_length,sampling_rate=1,
                                   length=length,batch_size=1000000)
    X , Y = data_gen[0]
    # split training, testing set
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.6,shuffle=True,random_state=7)
    print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
    dataDict ={'X_train':X_train,
          'X_test':X_test,
          'y_train':Y_train,
          'y_test':Y_test}

    return dataDict