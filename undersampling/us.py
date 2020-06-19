from __future__ import absolute_import
from sklearn.model_selection import StratifiedKFold

import numpy as np
import os
import abc
import sys

sys.path.append('..')
from opf.models.supervised import SupervisedOPF

class US(metaclass=abc.ABCMeta):

	def __init__(self):
		self.min_class_label = None

	def __computeScore(self, labels, preds, conqs, score):
		
		for i in range(len(labels)):
		    if labels[i]==preds[i]:
		        score[conqs[i]]+=1
		    else:
		        score[conqs[i]]-=1

	def __runOPF(self, X_train,y_train,index_train,X_test,y_test,index_test, score):
		# Creates a SupervisedOPF instance
		opf = SupervisedOPF(distance='log_squared_euclidean',
		                    pre_computed_distance=None)

		# Fits training data into the classifier
		opf.fit(X_train, y_train, index_train)
		
		# Predicts new data
		preds, conqs = opf.predict(X_test)
		
		self.__computeScore(y_test, preds, conqs, score)

	def run(self, X, Y):
		indices = np.arange(len(X))
		
		# Counting the number of samples in each class
		(uniques, frequency) = np.unique(Y, return_counts=True)
		
		# Indices of the minority and majority classes
		idx_min = np.argmin(frequency)
		idx_max = np.argmax(frequency)
		
		# Number of samples 
		minority_class, majority_class = uniques[idx_min], uniques[idx_max]        
		
		# Create stratified k-fold subsets
		kfold = 5 # no. of folds
		skf = StratifiedKFold(kfold, shuffle=True,random_state=1)
		skfind = [None] * kfold  # skfind[i][0] -> train indices, skfind[i][1] -> test indices
		cnt = 0
		for index in skf.split(X, Y):
		    skfind[cnt] = index
		    cnt += 1        
		
		score = np.zeros((5,len(X)))

		for i in range(kfold):
		    train_indices = skfind[i][0]   
		    test_indices = skfind[i][1]
		    X_train = X[train_indices]
		    y_train = Y[train_indices]
		    index_train = indices[train_indices]
		
		
		    X_test = X[test_indices]
		    y_test = Y[test_indices]
		    index_test = indices[test_indices]
		    self.__runOPF(X_train,y_train,index_train,X_test,y_test,index_test, score[i])
		

		output=  np.zeros((len(indices),8))

		score_t = np.transpose(score)
		output[:,0] =indices
		output[:,1] =Y
		output[:,2] =np.sum(score_t,axis=1)
		output[:,3:] =score_t

		return output, majority_class, minority_class

	def fit_resample(self, X, y, valid = None):
		"""Fits the model and returns a new dataset.

		# Arguments
			X: The feature matrix with shape `[n_samples, n_features]`.
			y: Labels `[n_samples]`.
			valid: Validation feature `[v_samples, n_features]`. 
				Used to remove the validation samples from X,
				for cases when X = Training + Validation sets.

		# Return
			X_resamp: Resampled data.
			y_resamp: Resampled data labels.
		"""
		output, majority_class, self.min_class_label = self.run(X, y)
		if not valid is None:
			X = X[:-len(valid),...]
			y = y[:-len(valid),...]
			output = output[:-len(valid),...]
		X_resamp, y_resamp =  self.variant(output, X, y, majority_class, self.min_class_label) 
		return X_resamp, y_resamp

	@abc.abstractmethod
	def variant(self, output, X, Y,  majority_class, minority_class):
		return


