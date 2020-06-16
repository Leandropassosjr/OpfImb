from typing import List, Tuple, Dict, Optional
from collections import defaultdict

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score, recall_score, f1_score
import sys
import abc

sys.path.append('..')
from opf.models.unsupervised import UnsupervisedOPF
from opf.models.supervised import SupervisedOPF


from oversampling.core.gp import GaussianParams
from oversampling.core import estimation
from oversampling.core import generation



CLUSTER_MIN_SIZE = 2


class OS(metaclass=abc.ABCMeta):
	def __init__(self):
		self.opfSup = SupervisedOPF(distance='log_squared_euclidean', pre_computed_distance=None)


	def __classify(self, x_train,y_train, x_valid, y_valid, minority_class):
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

	def run(self,
			X: np.ndarray,
			max_k: int = 5,
			d = None) -> Tuple[UnsupervisedOPF, Dict[int, List[int]]]: 

		"""Run unsupervised OPF.

		# Arguments
			X: The feature matrix with shape `[n_samples, n_features]`.
			max_k: Unsupervised OPF hyperparameter.

		# Return
			clf: Fitted unsupervised OPF classifier.
			cluster2samples: Maps the i-th cluster to a list of all its samples.
		"""
		clusterer = UnsupervisedOPF(max_k=max_k, pre_computed_distance=d)
		clusterer.fit(X)

		# Keep track of all samples in each cluster

		cluster2samples = defaultdict(list)
		for idx, sample in enumerate(clusterer.subgraph.nodes):
			cluster2samples[sample.cluster_label].append(idx)

		# The list i-th element corresponds to the i-th cluster
		clusterer.fit(X)

		# Keep track of all samples in each cluster
		cluster2samples = defaultdict(list)
		for idx, sample in enumerate(clusterer.subgraph.nodes):
			cluster2samples[sample.cluster_label].append(idx)
	
		return clusterer, dict(cluster2samples)


	def computeVariant(self, 
			 clf: UnsupervisedOPF, 
			 cluster2samples: Dict[int, List[int]],
			 X: np.ndarray,
			 generate_n: int,
			 estimation_fn: callable = estimation.mean_gaussian,
			 sampling_fn: callable = generation.sampling) -> List[np.ndarray]:
		cluster2samples = sorted(cluster2samples.items(), key=lambda entry: entry[0])
		gaussian_params = []

		for counter, (cluster_id, sample_ids) in enumerate(cluster2samples):

			if len(sample_ids) < CLUSTER_MIN_SIZE:
				continue

			mean, cova = estimation_fn(X, sample_ids, clf.subgraph)
			gaussian_params.append(GaussianParams(
				sample_ids,
				mean,
				cova,
			))

		# Consider only samples in valid clusters to compute fraction of samples
		# per cluster, else the number of generated samples may be smaller than
		# `generate_n`
		available_samples = sum(
			len(s)
			for _, s in cluster2samples
			if len(s) >= CLUSTER_MIN_SIZE
		)

		new_samples = []
		for i, gp in enumerate(gaussian_params):
			n_new_samples = int(np.round(generate_n * gp.n / available_samples))
			if n_new_samples < 1:
				continue

			new_samples.append(sampling_fn(
				gp,
				n_new_samples,
				X[gp.sample_ids]
			))

		return new_samples

	def saveResults(self, X_train,Y_train, X_test, Y_test,  ds,f, approach, minority_class, exec_time, path_output):

		path = '{}/{}/{}/{}'.format(path_output,approach,ds,f)
		if not os.path.exists(path):
			os.makedirs(path)

		results_print=[]
		accuracy, recall, f1, pred = self.__classify(X_train,Y_train, X_test, Y_test, minority_class)
		results_print.append([0,accuracy, recall, f1, exec_time])

		np.savetxt('{}/pred.txt'.format(path), pred, fmt='%d')
		np.savetxt('{}/results.txt'.format(path), results_print, fmt='%d,%.5f,%.5f,%.5f,%.5f')

	def saveDataset(self, X_train,Y_train, pathDataset,approach):
		DS = np.insert(X_train,len(X_train[0]),Y_train , axis=1)
		np.savetxt('{}/train_{}.txt'.format(pathDataset, approach),DS,  fmt='%.5f,'*(len(X_train[0]))+'%d')    

	@abc.abstractmethod
	def variant(self, output, X, Y,  majority_class, minority_class):
		return

	@abc.abstractmethod
	def fit_resample(self, X, y):
		return
