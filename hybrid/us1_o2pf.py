# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 19:43:35 2020

@author: DANILO
"""

import sys
sys.path.append('..')

from undersampling.opf_us1 import OpfUS1
from oversampling.o2pf import O2PF
from oversampling.core import arrays
from collections import Counter

class US1O2pf:
    def __init__(self, k_max):
        self.k_max = k_max
        '''
        '''
    
    def fit_resample(self, X, y):
        opf_us1 = OpfUS1()
        o2pf_ = O2PF()
        
        label_freq = Counter(y)
        min_label, min_freq = label_freq.most_common()[-1]
        _, max_freq = label_freq.most_common()[0]
        
        # Split minority from majority classes
        min_x, min_y, max_x, max_y = arrays.separate(X, y, min_label)        
        
        n_samples_generate = max_freq - min_freq
        
        X_res, y_res = opf_us1.fit_resample(X, y) # Undersampling
        X_res_synth = o2pf_.fit_resample(X_res, n_samples_generate, self.k_max) # Oversampling
    
        # Concatenate
        over_x, over_y = arrays.join_synth(min_x, min_y, X_res_synth, min_label)
        all_x, all_y = arrays.concat_shuffle([max_x, over_x], [max_y, over_y])        
        
        return all_x, all_y