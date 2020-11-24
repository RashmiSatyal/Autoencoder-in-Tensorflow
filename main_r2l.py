# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 23:22:46 2020

@author: Rashmi
"""


from __future__ import absolute_import, division, print_function,unicode_literals

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.metrics import confusion_matrix,accuracy_score,recall_score,precision_score,f1_score

import numpy as np
import tensorflow as tf
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation

from preprocess import *
from featurepreparation import *
from model import *
from predictions import *
from plot import *
#from featureselection import *


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

print('NSL-KDD dataset preprocessor', end='\n\n')
service_list = get_service_list(dirname='list', filename='service.txt')
flag_list = get_flag_list(dirname='list', filename='flag.txt')

#remove column label from training and test
train_data = pd.read_csv('D:/NDSU/Research/Project/NSL-KDD/KDDTrain+.txt',
                         header=None, names = col_names, index_col = False)
test_data = pd.read_csv('D:/NDSU/Research/Project/NSL-KDD/KDDTest+.txt',
                        header=None, names = col_names, index_col = False)

training_outcome = train_data.label.unique()
print("The training set has {} possible outcomes \n"
      .format(len(training_outcome)) )
print(sorted(training_outcome))

test_outcome = test_data.label.unique()
print("The test set has {} possible outcomes \n".format(len(test_outcome)) )


train_data = to_numeric(train_data, service_list, flag_list)
test_data = to_numeric(test_data, service_list, flag_list)

#print(train_data_processes)



dos_attacks=["snmpgetattack","back","land","neptune","smurf","teardrop","pod",
             "apache2","udpstorm","processtable","mailbomb"]
r2l_attacks=["snmpguess","worm","httptunnel","named","xlock","xsnoop",
             "sendmail","ftp_write","guess_passwd","imap","multihop","phf",
             "spy","warezclient","warezmaster"]
u2r_attacks=["sqlattack","buffer_overflow","loadmodule","perl","rootkit",
             "xterm","ps"]
probe_attacks=["ipsweep","nmap","portsweep","satan","saint","mscan"]

# New labels 
classes=["Normal","Dos","R2L","U2R","Probe"]

#TODO: label_attack in a different file
#TODO: Print out the labels for attacks from label_attack module

#Helper function to label samples to 5 classes
def label_attack (row):
    if row["label"] in dos_attacks:
        return classes[1]
    if row["label"] in r2l_attacks:
        return classes[2]
    if row["label"] in u2r_attacks:
        return classes[3]
    if row["label"] in probe_attacks:
        return classes[4]
    return classes[0]

#Combine the datasets temporarily to do the labeling 
test_samples_length = len(test_data)
df=pd.concat([train_data,test_data])
df["Class"]=df.apply(label_attack,axis=1)

#TODO: Scale continuous values
#TODO: Choose encoding method

# The old outcome field is dropped since it was replaced with the Class field,
# the difficulty field will be dropped as well.
df=df.drop("label",axis=1)


# we again split the data into training and test sets.
train_data= df.iloc[:-test_samples_length, :]
test_data= df.iloc[-test_samples_length:,:]


sympolic_columns=["protocol_type","service","flag"]
label_column="Class"
for column in df.columns :
    if column in sympolic_columns:
        encode_text(train_data,test_data,column)
    elif not column == label_column:
        minmax_scale_values(train_data,test_data, column)

train_header = train_data.head()
test_header = test_data.head()
#TODO: convert object arrays
#TODO: each df should contain not only data from the specific attack but also
#normal data

R2LClass = ['R2L', 'Normal']
R2L_df_train = train_data[train_data.Class.isin(R2LClass)]

R2L_df_test = test_data[test_data.Class.isin(R2LClass)]

# =============================================================================
# 
# Feature scaling
# 
# =============================================================================
# Split dataframes into X & Y
# assign X as a dataframe of feautures and Y as a series of outcome variables

X_R2L = R2L_df_train.drop('Class', 1)
Y_R2L = R2L_df_train.Class


X_R2L_test = R2L_df_test.drop('Class', 1)
Y_R2L_test = R2L_df_test.Class


columns=list(X_R2L)
columns_test=list(X_R2L_test)

#Scale df using StandardScalar()
scaler1 = preprocessing.StandardScaler().fit(X_R2L)
X_R2L=scaler1.transform(X_R2L) 
# test data
scaler5 = preprocessing.StandardScaler().fit(X_R2L_test)
X_R2L_test=scaler5.transform(X_R2L_test) 

X_R2L_test = pd.DataFrame(X_R2L_test)
#Feature selection based on attack types

#DoS attack features
#X_new_DoS = X_newDoS(X_DoS,Y_DoS)
#newcolname_DoS = newcolname_DoS(columns)

#np.seterr(divide='ignore', invalid='ignore')
selector = SelectPercentile(f_classif, percentile=10)

X_newR2L = selector.fit_transform(X_R2L,Y_R2L)



true=selector.get_support()
newcolindex_R2L = [i for i, x in enumerate(true) if x]
newcolname_R2L = list(columns[i] for i in newcolindex_R2L)


X_R2L_test = X_R2L_test[X_R2L_test.columns[newcolindex_R2L]]


#Training R2L
xr2l,yr2l=X_newR2L,Y_R2L
#xdos=xdos.values
x_testr2l,y_testr2l=X_R2L_test,Y_R2L_test
#x_testdos=x_testdos.values
y0r2l=np.ones(len(yr2l),np.int8)
y0r2l[np.where(yr2l==classes[0])]=0
y0_testr2l=np.ones(len(y_testr2l),np.int8)
y0_testr2l[np.where(y_testr2l==classes[0])]=0

autoencoder = getModel(xr2l)
history=autoencoder.fit(xr2l[np.where(y0r2l==0)],xr2l[np.where(y0r2l==0)],
               epochs=10,
                batch_size=100,
                shuffle=True,
                validation_split=0.1
                       )

print(type(X_R2L_test))

# We set the threshold equal to the training loss of the autoencoder
threshold=history.history["loss"][-1]
x_testr2l=x_testr2l.to_numpy()
testing_set_predictions=autoencoder.predict(x_testr2l)
test_losses=calculate_losses(x_testr2l,testing_set_predictions)
testing_set_predictions=np.zeros(len(test_losses))
testing_set_predictions[np.where(test_losses>threshold)]=1


accuracy=accuracy_score(y0_testr2l,testing_set_predictions)

#inline plot
c = confusion_matrix(y0_testr2l,testing_set_predictions)
plot_confusion_matrix(c,["Normal","R2L"])
#violin_plot(["Normal","Attack"], test_losses, y_test, threshold)