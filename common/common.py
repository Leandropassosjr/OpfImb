from sklearn.metrics import accuracy_score, recall_score, f1_score
from opf.models.supervised import SupervisedOPF

import numpy as np
import os

class COMMON():
	def __init__(self):
		self.opfSup = SupervisedOPF(distance='log_squared_euclidean', pre_computed_distance=None)

	def classify(self, x_train,y_train, x_valid, y_valid, minority_class):
		# Training the OPF                
		indexes = np.arange(len(x_train))
		self.opfSup.fit(x_train, y_train,indexes)

		# Prediction of the validation samples
		y_pred,_ = self.opfSup.predict(x_valid)
		y_pred = np.array(y_pred)
		
		# Validation measures for this k nearest neighbors
		accuracy = accuracy_score(y_valid, y_pred)
		recall = recall_score(y_valid, y_pred, pos_label=minority_class) # assuming that 2 is the minority class
		f1 = f1_score(y_valid, y_pred, pos_label=minority_class)
		return accuracy, recall, f1, y_pred


	def saveResults(self, X_train,Y_train, X_test, Y_test,  ds,f, approach, minority_class, exec_time, path_output):

		path = '{}/{}/{}/{}'.format(path_output,approach,ds,f)
		if not os.path.exists(path):
		    os.makedirs(path)

		results_print=[]
		accuracy, recall, f1, pred = self.classify(X_train,Y_train, X_test, Y_test, minority_class)
		results_print.append([0,accuracy, recall, f1, exec_time])

		np.savetxt('{}/pred.txt'.format(path), pred, fmt='%d')
		np.savetxt('{}/results.txt'.format(path), results_print, fmt='%d,%.5f,%.5f,%.5f,%.5f')
		print('Results:')
		print('    {}/results.txt'.format(path))

	def saveDataset(self, X_train,Y_train, pathDataset,approach):
		DS = np.insert(X_train,len(X_train[0]),Y_train , axis=1)
		np.savetxt('{}/train_{}.txt'.format(pathDataset, approach),DS,  fmt='%.5f,'*(len(X_train[0]))+'%d')   
		print('Dataset generated:')
		print('    {}/train_{}.txt'.format(pathDataset, approach))

