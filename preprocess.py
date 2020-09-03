# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 10:30:38 2020

@author: Rashmi
"""

import os
import pandas as pd
import numpy as np

#import h5py

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.utils import shuffle

def get_service_list(dirname = 'list', filename = 'service.txt'):
    print('Getting service list.....')
    with open(os.path.join(dirname, filename), 'r') as service:
        service_list = service.read().split('\n')
    return service_list

def get_flag_list(dirname = 'list', filename = 'flag.txt'):
    print('Getting flag list.....')
    with open(os.path.join(dirname, filename), 'r') as flag:
        flag_list = flag.read().split('\n')
    return flag_list

def get_data_frame(dirname = 'dataset', filename = None):
    if filename is None:
        raise ValueError('Filename should be provided.')
    print('Getting dataframe from file....')
    print('Read file: ', os.path.join(dirname, filename))
    df = pd.read_csv(os.path.join(dirname, filename), header=None)
    return df

def to_numeric(data_frame, service_list, flag_list, test = False, attack = False, save = False):
    df = data_frame
    # index 1: protocol_type
    print('Converting protocol_type values to numeric....')

    df['protocol_type'].replace(['tcp', 'udp', 'icmp'], range(3), inplace=True)
    
    # index 2: service
    print('Replacing service values to numeric...')
    df['service'].replace(service_list, range(len(service_list)), inplace=True)

    # index 3: flag
    print('Replacing flag values to numeric...')
    df['flag'].replace(flag_list, range(len(flag_list)), inplace=True)
#feature extraction
#    if not test:
#        # extract only the same features from Kyoto 2006+ dataset
#        df = df.loc[:, [0, 1, 2, 3, 4, 5, 22, 24, 25, 28, 31, 32, 35, 37, 38]]
#    else:
#        # include label
#        df = df.loc[:, [0, 1, 2, 3, 4, 5, 22, 24, 25, 28, 31, 32, 35, 37, 38, 41]]
#        df[41] = df[41].map(lambda x: 0 if x == 'normal' else 1)  # normal 0, attack 1
    
    # save as csv file
    if save:
        if not os.path.exists('csv'):
            os.makedirs('csv')
        if not test:
            if not attack:
                print('Saving file:', os.path.join('csv', 'train_normal_numeric.csv'))
                df.to_csv(os.path.join('csv', 'train_normal_numeric.csv'))
            else:
                print('Saving file:', os.path.join('csv', 'train_mixed_numeric.csv'))
                df.to_csv(os.path.join('csv', 'train_mixed_numeric.csv'))
        else:
            print('Saving file:', os.path.join('csv', 'test_numeric.csv'))
            df.to_csv(os.path.join('csv', 'test_numeric.csv'))

    return df