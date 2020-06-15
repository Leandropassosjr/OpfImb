# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 17:14:53 2020

@author: DANILO
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 15:10:02 2020

@author: DANILO
"""
from undersampling.us import US
import numpy as np

class OpfUS4(US):
    
    def variant(self, output, X, Y, majority_class, minority_class):
        #4st case: remove samples from majoritary class with negative or zero scores 
        # and from minoritary class with negative scores
        output_negatives = output[output[:,2]<0]

        output_negatives_major_zero = output_negatives[output_negatives[:,1]==majority_class]
        output_negatives_major_zero = output_negatives_major_zero[output_negatives_major_zero[:,2]<=0]

        X_train = np.delete(X, output_negatives_major_zero[:,0],0)
        y_train = np.delete(Y, output_negatives_major_zero[:,0])
        
        return X_train, y_train
        
    def fit_resample(self, X, y):
        output, majority_class, minority_class = self.run(X, y)
        X_res, y_res = self.variant(output, X, y, majority_class, minority_class)
        
        return X_res, y_res