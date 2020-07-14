from hybrid.us1_o2pf import US1O2PF
from hybrid.us2_o2pf import US2O2PF
from hybrid.us3_o2pf import US3O2PF
from common.common import COMMON

import os
import numpy as np
from time import time


def perform_under(**kwargs):
	# Apply the hybrid approach according to the opf-us variant represented by the hybrid_obj
	hybrid_obj = kwargs['hybrid_obj']
	X = kwargs['X']
	y = kwargs['y']
	X_test = kwargs['X_test']
	y_test = kwargs['y_test']
	f = kwargs['fold']
	ds = kwargs['ds']
	valid = kwargs['valid'] 

	start_time = time()
	all_x, all_y = hybrid_obj.fit_resample(X, y)
	end_time = time() - start_time

	approach = hybrid_obj.__class__.__name__ 

	common = COMMON()

	# Save the results of the oversampling
	common.saveTimeOnly(ds, f, approach, end_time, 'Results')


datasets = ['vertebral_column']
#paper uses 20 folds, so the next line runs as follows:
#folds = np.arange(1,21)
folds = np.arange(1,2)

# Objects for hybrid approach
us1_o2pf = US1O2PF()
us2_o2pf = US2O2PF()
us3_o2pf = US3O2PF()

for dsds in range(len(datasets)):
	ds = datasets[dsds]
	for ff in range(len(folds)): 
		f = folds[ff]
		train = np.loadtxt('data/{}/{}/train.txt'.format(ds,f),delimiter=',', dtype=np.float32)
		valid = np.loadtxt('data/{}/{}/valid.txt'.format(ds,f),delimiter=',', dtype=np.float32)
		test = np.loadtxt('data/{}/{}/test.txt'.format(ds,f),delimiter=',', dtype=np.float32)
		
		concat = np.concatenate((train, valid))
		X = concat[:,:-1]
		Y = concat[:,-1].astype(np.int) 
			    
		X_test = test[:,:-1]
		Y_test = test[:,-1].astype(np.int)
			    
		pathDataset = 'data/{}/{}'.format(ds,f)
		if not os.path.exists(pathDataset):
			os.makedirs(pathDataset)   

		#1st variant: remove samples from majority class with negative scores  
		perform_under(hybrid_obj=us1_o2pf, X=X, y=Y, X_test=X_test, y_test=Y_test, fold=f, ds=ds,valid=valid)          

	   	#2st variant: remove samples from majority class with negative or zero scores
		perform_under(hybrid_obj=us2_o2pf,  X=X, y=Y, X_test=X_test, y_test=Y_test, fold=f, ds=ds,valid=valid)         
	
		#3st variant: remove all samples with negative
		perform_under(hybrid_obj=us3_o2pf, X=X, y=Y, X_test=X_test, y_test=Y_test, fold=f, ds=ds,valid=valid)   
