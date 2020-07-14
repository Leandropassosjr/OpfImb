from typing import List, Tuple, Dict, Optional
from collections import defaultdict
from collections import Counter
from sklearn.utils import shuffle

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import sys
import abc

sys.path.append('..')
from opf.models.unsupervised import UnsupervisedOPF

from oversampling.core.gp import GaussianParams
from oversampling.core import estimation
from oversampling.core import generation
from oversampling.core import arrays

CLUSTER_MIN_SIZE = 2


class OS(metaclass=abc.ABCMeta):
	def __init__(self, k_max=5):
		self.min_class_label = None
		self.k_max = k_max

	def run(self,
			X: np.ndarray,
			d = None) -> Tuple[UnsupervisedOPF, Dict[int, List[int]]]: 

		"""Run unsupervised OPF.

		# Arguments
			X: The feature matrix with shape `[n_samples, n_features]`.

		# Return
			clf: Fitted unsupervised OPF classifier.
			cluster2samples: Maps the i-th cluster to a list of all its samples.
		"""

		clusterer = UnsupervisedOPF(max_k=self.k_max, pre_computed_distance=d)
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
		max_gp_n = 0
		max_gp_index = -1
		for i, gp in enumerate(gaussian_params):		
			if max_gp_n<gp.n:
				max_gp_n = gp.n
				max_gp_index = i
	
			n_new_samples = int(np.round(generate_n * gp.n / available_samples))
			if n_new_samples < 1:
				continue

			new_samples.append(sampling_fn(
			    gp,
			    n_new_samples,
			    X[gp.sample_ids]
			))
		diff = generate_n-len(new_samples)		
		if diff>0:
			gp = gaussian_params[max_gp_index]
			new_samples.append(sampling_fn(
			    gp,
			    diff,
			    X[gp.sample_ids]
			))			

		return new_samples

	def splitMinorityDataset(self, X, y):
		label_freq = Counter(y)
		min_label, min_freq = label_freq.most_common()[-1]
		_, max_freq = label_freq.most_common()[0]

		# Split minority from majority classes
		min_x, min_y, max_x, max_y = arrays.separate(X, y, min_label)

		# Oversample
		min_x, min_y = shuffle(min_x, min_y)
		n_new_samples = max_freq - min_freq
		return min_x, min_y, max_x, max_y, min_label, n_new_samples


	def concatenate(self, min_x, min_y, max_x, max_y, synth_x, min_label):
		# Make a list of ndarrays into a single ndarray
		synth_x = np.concatenate(synth_x)

		# Concatenate
		over_x, over_y = arrays.join_synth(min_x, min_y, synth_x, min_label)
		all_x, all_y = arrays.concat_shuffle([max_x, over_x], [max_y, over_y])
		return all_x, all_y

	def fit_resample(self, X, y):
		min_x, min_y, max_x, max_y, self.min_class_label, n_new_samples = self.splitMinorityDataset( X, y)
		synth_x = self.variant(min_x, n_new_samples)
		return self.concatenate(min_x, min_y, max_x, max_y, synth_x, self.min_class_label)

	@abc.abstractmethod
	def variant(self, output, X, Y,  majority_class, minority_class):
		return
