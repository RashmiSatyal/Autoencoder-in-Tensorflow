# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 21:21:23 2020

@author: Rashmi
"""

# =============================================================================
# 
# Feature Selection
#
# =============================================================================

from sklearn.feature_selection import SelectPercentile, f_classif

#univariate feature selection with ANOVA F-test. using secondPercentile method, then RFE
#Scikit-learn exposes feature selection routines as objects that implement the transform method
#SelectPercentile: removes all but a user-specified highest scoring percentage of features
#f_classif: ANOVA F-value between label/feature for classification tasks.

def X_newDoS(X_DoS,Y_DoS):
    np.seterr(divide='ignore', invalid='ignore')
    selector = SelectPercentile(f_classif,percentile=10)
    return(selector.fit_transform(X_DoS,Y_DoS))
    
def newcolname_DoS(columns):
    selector = SelectPercentile(f_classif,percentile=10)
    true = selector.get_support
    newcolindex_DoS = [i for i, x in enumerate(true) if x]
    return(list( columns[i] for i in newcolindex_DoS )) 
    
