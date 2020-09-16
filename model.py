# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 14:44:14 2020

@author: Rashmi
"""
import tensorflow as tf
from keras.layers import Input,Dropout,Dense
from keras.models import Model
from keras import regularizers
from keras.utils.data_utils import get_file
import numpy as np

#Model building and training

#TODO: pass x and y values to getModel
#x:feature
#y:classification
#y0 holds the classification of the training dataset to one of two possible 
    #labels, 0 for normal traffic or 1 for an attack
def getModel(x):
    inp = Input(shape=(x.shape[1],))
    d1=Dropout(0.2)(inp)
    encoded = Dense(10, activation='relu', activity_regularizer=regularizers.l2(10e-5))(d1)
    decoded = Dense(x.shape[1], activation='relu')(encoded)
    autoencoder = Model(inp, decoded)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    return autoencoder

