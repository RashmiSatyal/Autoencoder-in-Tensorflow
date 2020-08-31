# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 21:00:22 2020

@author: Rashmi
"""


from __future__ import absolute_import, division, print_function, unicode_literals

from sklearn.preprocessing import StandardScaler

import numpy as np
import tensorflow as tf
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation

from preprocess import *
from featurepreparation import *


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

print('NSL-KDD dataset preprocessor', end='\n\n')
service_list = get_service_list(dirname='list', filename='service.txt')
flag_list = get_flag_list(dirname='list', filename='flag.txt')

#remove column label from training and test
train_data = pd.read_csv('D:/NDSU/Research/Project/NSL-KDD/KDDTrain+.txt',header=None, names = col_names, index_col = False)
test_data = pd.read_csv('D:/NDSU/Research/Project/NSL-KDD/KDDTest+.txt',header=None, names = col_names, index_col = False)

training_outcome = train_data.label.unique()
print("The training set has {} possible outcomes \n".format(len(training_outcome)) )
#print(sorted(training_outcome))

test_outcome = test_data.label.unique()
print("The test set has {} possible outcomes \n".format(len(test_outcome)) )


train_data_processes = to_numeric(train_data, service_list, flag_list)

#print(train_data_processes)


dos_attacks=["snmpgetattack","back","land","neptune","smurf","teardrop","pod","apache2","udpstorm","processtable","mailbomb"]
r2l_attacks=["snmpguess","worm","httptunnel","named","xlock","xsnoop","sendmail","ftp_write","guess_passwd","imap","multihop","phf","spy","warezclient","warezmaster"]
u2r_attacks=["sqlattack","buffer_overflow","loadmodule","perl","rootkit","xterm","ps"]
probe_attacks=["ipsweep","nmap","portsweep","satan","saint","mscan"]

# New labels 
classes=["Normal","Dos","R2L","U2R","Probe"]

#TODO: label_attack in a different file
#TODO: Print out the labels for attacks from label_attack module

#Helper function to label samples to 5 classes
def label_attack (row):
    if row["outcome"] in dos_attacks:
        return classes[1]
    if row["outcome"] in r2l_attacks:
        return classes[2]
    if row["outcome"] in u2r_attacks:
        return classes[3]
    if row["outcome"] in probe_attacks:
        return classes[4]
    return classes[0]

#TODO: Scale continuous values
#TODO: Choose encoding method




#TODO: review code below (x holds features, y holds clasification)
    
#x,y=train_data,train_data.pop("Class").values
#x=x.values
#x_test,y_test=test_data,test_data.pop("Class").values
#x_test=x_test.values
#y0=np.ones(len(y),np.int8)
#y0[np.where(y==classes[0])]=0
#y0_test=np.ones(len(y_test),np.int8)
#y0_test[np.where(y_test==classes[0])]=0