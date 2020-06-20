from oversampling.o2pf import O2PF
from oversampling.o2pf_ri import O2PF_RI
from oversampling.o2pf_mi import O2PF_MI
from oversampling.o2pf_p import O2PF_P
from oversampling.o2pf_wi import O2PF_WI
from common.common import COMMON


import os
import numpy as np
import sys
from time import time

np.set_printoptions(threshold=sys.maxsize)

def perform_over(**kwargs):
	o2pf_obj = kwargs['o2pf_obj']
	X = kwargs['X']
	y = kwargs['y']
	X_test = kwargs['X_test']
	y_test = kwargs['y_test']
	ds = kwargs['ds']
	f = kwargs['f']

	start_time = time()
	all_x, all_y = o2pf_obj.fit_resample( X, y)
	end_time = time() -start_time

	approach = o2pf_obj.__class__.__name__

	common = COMMON()

	common.saveDataset(all_x, all_y, pathDataset, approach)
	common.saveResults(all_x, all_y, X_test, y_test, ds, f, approach, o2pf_obj.min_class_label, end_time, 'Results')


datasets = ['vertebral_column']
#paper uses 20 folds, so the next line runs as follows:
#folds = np.arange(1,21)
folds = np.arange(1,2)
#k_max = [5,10,20,30,40,50]
k_max = 5


# Objects for undersampling
o2pf = O2PF(k_max=k_max)
o2pf_ri = O2PF_RI()
o2pf_mi = O2PF_MI()
o2pf_p = O2PF_P()
o2pf_wi = O2PF_WI()

for dsds in range(len(datasets)):
	ds = datasets[dsds]
	for ff in range(len(folds)): 
		f = folds[ff]
		train = np.loadtxt('data/{}/{}/train.txt'.format(ds,f),delimiter=',', dtype=np.float32)
		test = np.loadtxt('data/{}/{}/test.txt'.format(ds,f),delimiter=',', dtype=np.float32)

		X = train[:,:-1]
		y = train[:,-1].astype(np.int) 

		X_test = test[:,:-1]
		y_test = test[:,-1].astype(np.int) 


		pathDataset = 'data/{}/{}'.format(ds,f)
		if not os.path.exists(pathDataset):
			os.makedirs(pathDataset)   

        #main approach: generate samples from minority class until balancing the dataset
		perform_over(o2pf_obj = o2pf, X = X,y = y, X_test = X_test, y_test = y_test,
			ds = ds, f = f)

		#o2pf_ri approach: 
		perform_over(o2pf_obj = o2pf_ri,  X = X,y = y, X_test = X_test, y_test = y_test,
			ds = ds, f = f)

		#o2pf_mi approach: 
		perform_over(o2pf_obj = o2pf_mi, X = X,y = y, X_test = X_test, y_test = y_test,
			ds = ds, f = f)

		#o2pf_p approach: 
		perform_over(o2pf_obj = o2pf_p,  X = X,y = y, X_test = X_test, y_test = y_test,
			ds = ds, f = f)

		#o2pf_wi approach: 
		perform_over(o2pf_obj = o2pf_wi,  X = X,y = y, X_test = X_test, y_test = y_test,
			ds = ds, f = f)
