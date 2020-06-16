from oversampling.o2pf import O2PF
from oversampling.o2pf_ri import O2PF_RI
from oversampling.o2pf_mi import O2PF_MI
from oversampling.o2pf_p import O2PF_P
from oversampling.o2pf_wi import O2PF_WI
from oversampling.core import arrays

import os
import numpy as np
import sys
from time import time
from collections import Counter
from sklearn.utils import shuffle

import logging
logging.disable(sys.maxsize)
np.set_printoptions(threshold=sys.maxsize)





def perform_over(**kwargs):
	o2pf_obj = kwargs['o2pf_obj']
	min_x = kwargs['min_x']
	min_y = kwargs['min_y']
	max_x = kwargs['max_x']
	max_y = kwargs['max_y']
	ds = kwargs['ds']
	f = kwargs['f']
	min_label = kwargs['min_label']
	n_new_samples = kwargs['n_new_samples']
	k_max = kwargs['k_max']

	start_time = time()

	synth_x = o2pf_obj.variant(min_x, n_new_samples, k_max)
	# Make a list of ndarrays into a single ndarray
	synth_x = np.concatenate(synth_x)

	# Concatenate
	over_x, over_y = arrays.join_synth(min_x, min_y, synth_x, min_label)
	all_x, all_y = arrays.concat_shuffle([max_x, over_x], [max_y, over_y])

	approach = o2pf_obj.__class__.__name__

	end_time = time() -start_time

	o2pf_obj.saveDataset(all_x, all_y, pathDataset, approach)
	o2pf_obj.saveResults(all_x, all_y, X_test, y_test, ds, f, approach, min_label, end_time, 'Results')


datasets = ['vertebral_column']
#paper uses 20 folds, so the next line runs as follows:
#folds = np.arange(1,21)
folds = np.arange(1,2)
k_max = 5

# Objects for undersampling
o2pf = O2PF()
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

		label_freq = Counter(y)
		min_label, min_freq = label_freq.most_common()[-1]
		_, max_freq = label_freq.most_common()[0]

		# Split minority from majority classes
		min_x, min_y, max_x, max_y = arrays.separate(X, y, min_label)

		# Oversample
		min_x, min_y = shuffle(min_x, min_y)
		n_new_samples = max_freq - min_freq

		pathDataset = 'data/{}/{}'.format(ds,f)
		if not os.path.exists(pathDataset):
			os.makedirs(pathDataset)   

        #main approach: generate samples from minority class until balancing the dataset
		perform_over(o2pf_obj = o2pf, min_x = min_x,min_y = min_y, max_x = max_x, max_y = max_y,
			ds = ds, f = f, min_label = min_label, n_new_samples=n_new_samples, k_max=k_max)

		#o2pf_ri approach: 
		perform_over(o2pf_obj = o2pf_ri, min_x = min_x,min_y = min_y, max_x = max_x, max_y = max_y,
			ds = ds, f = f, min_label = min_label, n_new_samples=n_new_samples, k_max=k_max)

		#o2pf_mi approach: 
		perform_over(o2pf_obj = o2pf_mi, min_x = min_x,min_y = min_y, max_x = max_x, max_y = max_y,
			ds = ds, f = f, min_label = min_label, n_new_samples=n_new_samples, k_max=k_max)

		#o2pf_p approach: 
		perform_over(o2pf_obj = o2pf_p, min_x = min_x,min_y = min_y, max_x = max_x, max_y = max_y,
			ds = ds, f = f, min_label = min_label, n_new_samples=n_new_samples, k_max=k_max)

		#o2pf_wi approach: 
		perform_over(o2pf_obj = o2pf_wi, min_x = min_x,min_y = min_y, max_x = max_x, max_y = max_y,
			ds = ds, f = f, min_label = min_label, n_new_samples=n_new_samples, k_max=k_max)
