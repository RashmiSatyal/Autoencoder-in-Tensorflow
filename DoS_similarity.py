# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 21:04:48 2020

@author: Rashmi
"""

from __future__ import absolute_import, division, print_function,unicode_literals

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.metrics import confusion_matrix,accuracy_score,recall_score,precision_score,f1_score
from sklearn.cluster import KMeans
from collections import Counter, defaultdict

import numpy as np
import tensorflow as tf
import pandas as pd
import category_encoders as ce
import matplotlib.pyplot as plt


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation

from preprocess import *
from featurepreparation import *
from model import *
from predictions import *
from plot import *

col_names = ["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label"]

#remove column label from training and test
train_data = pd.read_csv('D:/NDSU/Research/Project/NSL-KDD/KDDTrain+.txt',
                         header=None, names = col_names, index_col = False)
test_data = pd.read_csv('D:/NDSU/Research/Project/NSL-KDD/KDDTest+.txt',
                        header=None, names = col_names, index_col = False)

training_outcome = train_data.label.unique()
train_services = train_data["service"].nunique()
train_flags = train_data["flag"].nunique()
train_ptype = train_data["protocol_type"].nunique()

test_outcome = test_data.label.unique()

dos_attacks=["snmpgetattack","back","land","neptune","smurf","teardrop","pod",
             "apache2","udpstorm","processtable","mailbomb"]
r2l_attacks=["snmpguess","worm","httptunnel","named","xlock","xsnoop",
             "sendmail","ftp_write","guess_passwd","imap","multihop","phf",
             "spy","warezclient","warezmaster"]
u2r_attacks=["sqlattack","buffer_overflow","loadmodule","perl","rootkit",
             "xterm","ps"]
probe_attacks=["ipsweep","nmap","portsweep","satan","saint","mscan"]

DoS_train_df = train_data[train_data.label.isin(dos_attacks)]
DoS_test_df = test_data[test_data.label.isin(dos_attacks)]

subset_normal_df = DoS_train_df[DoS_train_df["label"] == "snmpgetattack"]

def label_value (row):
    if row["label"] == "normal":
        return 0
    if row["label"] == "snmpgetattack":
        return 1
    if row["label"] == "back":
        return 2
    if row["label"] == "land":
        return 3
    if row["label"] == "neptune":
        return 4
    if row["label"] == "smurf":
        return 5
    if row["label"] == "teardrop":
        return 6
    if row["label"] == "pod":
        return 7
    if row["label"] == "apache2":
        return 8
    if row["label"] == "udpstorm":
        return 9
    if row["label"] == "processtable":
        return 10
    return 11

DoS_train_df["label"]=DoS_train_df.apply(label_value,axis=1)

test_samples_length = len(DoS_test_df)

DoS_train_services = DoS_train_df["service"].nunique()
DoS_train_flags = DoS_train_df["flag"].nunique()
DoS_train_ptype = DoS_train_df["protocol_type"].nunique()

df=pd.concat([DoS_train_df,DoS_test_df])

DoS_train_df= df.iloc[:-test_samples_length, :]
DoS_test_df= df.iloc[-test_samples_length:,:]

sympolic_columns=["protocol_type","service","flag"]
label_column="label"
for column in df.columns :
    if column in sympolic_columns:
        encode_text(DoS_train_df,DoS_test_df,column)
    elif not column == label_column:
        minmax_scale_values(DoS_train_df,DoS_test_df, column)

num_rows = []
for i in range(2,8):
    print(i)   
    subset_df = DoS_train_df[DoS_train_df["label"] == i]
    num_rows.append(subset_df.shape[0])

    

x,y=DoS_train_df,DoS_train_df.pop("label").values
x=x.values
kmeans = KMeans(n_clusters=6, random_state=0).fit(x)
km_labels = kmeans.labels_
km_total_labels = np.unique(km_labels)

clusters_indices = defaultdict(list)
for index, c  in enumerate(kmeans.labels_):
    clusters_indices[c].append(index)

centroids  = kmeans.cluster_centers_



#Compare centorid with each data in test
#Add label to clusters 
#Assign clusters' label to test data
#https://stackoverflow.com/questions/47291025/extracting-centroids-using-k-means-clustering-in-python
#wcss=[]    
#wcss.append(kmeans.inertia_)
#
#plt.plot(range(1,12),wcss)
#plt.title('The Elbow Method Graph')
#plt.xlabel('Number of clusters')
#plt.ylabel('WCSS')
#plt.show()
#
