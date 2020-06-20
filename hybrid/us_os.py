from __future__ import absolute_import
from oversampling.o2pf import O2PF

import numpy as np
import abc

class HYBRID(metaclass=abc.ABCMeta):

	def __init__(self, k_max=5):
		self.min_class_label = None
		self.k_max = k_max

	def fit_resample(self, X, y, valid = None):
		us_object = self.variant()
		o2pf_ = O2PF(self.k_max)

		X_res, y_res = us_object.fit_resample( X, y, valid)
		self.min_class_label = us_object.min_class_label
		all_x, all_y = o2pf_.fit_resample( X_res, y_res)   
		
		return all_x, all_y

	@abc.abstractmethod
	def variant(self, output, X, Y,  majority_class, minority_class):
		return


