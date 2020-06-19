# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 19:59:22 2020

@author: DANILO
"""

import sys
sys.path.append('..')

from undersampling.opf_us2 import OpfUS2
from oversampling.o2pf import O2PF

import numpy as np

class US1O2pf:
    def __init__(self, k_max):
        self.k_max = k_max
        '''
        '''
    
    def fit_resample(self, X, y):
        opf_us1 = OpfUS2()
        o2pf = O2PF()
        
        # Counting the number of samples in each class
        (uniques, frequency) = np.unique(y, return_counts=True)        
        
        # Indices of the minority and majority classes
        idx_min = np.argmin(frequency)
        idx_max = np.argmax(frequency)
        
        # Number of samples 
        minority_class, majority_class = uniques[idx_min], uniques[idx_max]
        
        n_samples_generate = majority_class - minority_class
        
        X_res, y_res = opf_us1.fit_resample(X, y) # Undersampling
        X_res, y_res = o2pf.fit_resample(X_res, n_samples_generate, self.k_max) # Oversampling
        
        return X_res, y_res