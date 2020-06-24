# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 21:00:22 2020

@author: Rashmi
"""


from __future__ import absolute_import, division, print_function, unicode_literals
import functools

import numpy as np
import tensorflow as tf
import pandas as pd


col_names = ["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label"]


train_data = pd.read_csv('D:/NDSU/Research/Project/NSL-KDD/KDDTrain+.txt',header=None, names = col_names)
test_data = pd.read_csv('D:/NDSU/Research/Project/NSL-KDD/KDDTest+.txt',header=None, names = col_names)


print (train_data.head(5))
# shape, this gives the dimensions of the dataset
print('Dimensions of the Training set:',train_data.shape)
print('Dimensions of the Test set:',test_data.shape)

#label distribution in training set
print('Label distribution Training set:')
print(train_data['label'].value_counts())
print()
print('Label distribution Test set:')
print(test_data['label'].value_counts())
