from __future__ import absolute_import
from oversampling.o2pf import O2PF

import numpy as np
from common.common import COMMON
import abc

class HYBRID(metaclass=abc.ABCMeta):
    def __init__(self):
        self.min_class_label = None

    def fit_resample(self, X, y, k_max, valid = None):
        us_object = self.variant()
        o2pf_ = O2PF()

        X_res, y_res = us_object.fit_resample( X, y, valid)
        self.min_class_label = us_object.min_class_label

        if valid is not None:
            X_valid = valid[:,:-1]
            y_valid = valid[:,-1].astype(np.int) 
            common = COMMON()
            best_k = common.optimization(X, y, X_valid, y_valid, o2pf_, k_max)
        else:
            best_k = k_max[0]

        all_x, all_y = o2pf_.fit_resample( X_res, y_res, best_k)   

        return all_x, all_y

    @abc.abstractmethod
    def variant(self, output, X, Y,  majority_class, minority_class):
        return



