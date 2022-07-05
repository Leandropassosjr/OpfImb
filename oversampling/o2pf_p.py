# -*- coding: utf-8 -*-
"""
Created on Mon Jun  15 17:01:00 2020

@author: LEANDRO
"""
from oversampling.os import OS
import numpy as np


from oversampling.core import estimation
from oversampling.core import generation

class O2PF_P(OS):
    
	def variant(self, X, generate_n, max_k):
		clf, cluster2samples = self.run(X, max_k)
		return self.computeVariant(clf, cluster2samples, X, generate_n,estimation.prototype_gaussian,generation.sampling)
