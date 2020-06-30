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

class OpfUS3(US):
    
	def variant(self, output, X, Y, majority_class, minority_class):
		#3st case: remove all samples with negative
		output_negatives = output[output[:,2]<0].astype(int)

		X_train = np.delete(X, output_negatives[:,0],0)
		y_train = np.delete(Y, output_negatives[:,0])
		
		return X_train, y_train
