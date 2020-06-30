# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 15:10:02 2020

@author: DANILO
"""
from undersampling.us import US
import numpy as np

class OpfUS1(US):
    
	def variant(self, output, X, Y,  majority_class, minority_class):
		#1st case: remove samples from majoritary class with negative scores        
		output_majority = output[output[:,1]==majority_class]
		output_majority_negative = output_majority[output_majority[:,2]<0].astype(int)

		X_train = np.delete(X, output_majority_negative[:,0],0)
		y_train = np.delete(Y, output_majority_negative[:,0])
		
		return X_train, y_train        

