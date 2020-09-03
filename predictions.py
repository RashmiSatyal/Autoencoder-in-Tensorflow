# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 10:59:35 2020

@author: Rashmi
"""
import numpy as np
# Helper function that calculates the reconstruction loss of each data sample
def calculate_losses(x,preds):
    losses=np.zeros(len(x))
    for i in range(len(x)):
        losses[i]=((preds[i] - x[i]) ** 2).mean(axis=None)
        
    return losses

