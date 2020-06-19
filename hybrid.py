from oversampling.o2pf import O2PF
from oversampling.core import arrays

from undersampling.opf_us1 import OpfUS1
from undersampling.opf_us2 import OpfUS2
from undersampling.opf_us3 import OpfUS3

from collections import Counter
from sklearn.utils import shuffle

import os
import numpy as np
import sys
from time import time

import logging
logging.disable(sys.maxsize)
np.set_printoptions(threshold=sys.maxsize)

def perform_under(**kwargs):
    # Apply the undersampling according to the opf-us variant represented by the opf_us_obj
    opf_us_obj = kwargs['us_obj']
    output = kwargs['output']
    X = kwargs['X']
    y = kwargs['y']
    X_test = kwargs['X_test']
    y_test = kwargs['y_test']
    ds = kwargs['ds']
    f = kwargs['fold']
    majority_class = kwargs['majority_class']
    minority_class = kwargs['minority_class']
    exec_time = kwargs['exec_time']
    k_max = kwargs['k_max']
    
    o2pf_obj = O2PF()    
    
    start_time = time()
    # Undersampling
    X_res, y_res = opf_us_obj.variant(output, X, y, majority_class, minority_class)
    end_time = time() - start_time + exec_time    
    
    approach = opf_us_obj.__class__.__name__
    
    # Save the results of the undersampling   
    opf_us_obj.saveDataset(X_res, y_res, pathDataset, approach)
    opf_us_obj.saveResults(X_res, y_res, X_test, y_test, ds, f, approach, minority_class, end_time, 'Results')
    
    start_time = time()
    # Get the minority and majority class labels
    label_freq = Counter(y_res)
    min_label, min_freq = label_freq.most_common()[-1]
    _, max_freq = label_freq.most_common()[0]
    
    # Split minority from majority classes
    min_x, min_y, max_x, max_y = arrays.separate(X_res, y_res, min_label)
    
    # Oversample
    min_x, min_y = shuffle(min_x, min_y)
    n_new_samples = max_freq - min_freq 
    
    synth_x = o2pf_obj.variant(min_x, n_new_samples, k_max)
    # Make a list of ndarrays into a single ndarray
    synth_x = np.concatenate(synth_x)
    
    # Concatenate
    over_x, over_y = arrays.join_synth(min_x, min_y, synth_x, min_label)
    all_x, all_y = arrays.concat_shuffle([max_x, over_x], [max_y, over_y])    
    
    end_time = time() - start_time + exec_time
    
    approach = opf_us_obj.__class__.__name__ + '_' + o2pf_obj.__class__.__name__
    
    # Save the results of the oversampling
    opf_us_obj.saveDataset(all_x, all_y, pathDataset, approach)
    opf_us_obj.saveResults(all_x, all_y, X_test, y_test, ds, f, approach, minority_class, end_time, 'Results')


datasets = ['vertebral_column']
#paper uses 20 folds, so the next line runs as follows:
#folds = np.arange(1,21)
folds = np.arange(1,2)
k_max = 5

# Objects for undersampling
us1 = OpfUS1()
us2 = OpfUS2()
us3 = OpfUS3()

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
        
        start_time = time()
        output, majority_class, minority_class = us1.run(X, Y)
        end_time = time() -start_time
        
        X = X[:len(train),...]
        Y = Y[:len(train),...]
        output = output[:len(train),...]
        
        X_test = test[:,:-1]
        Y_test = test[:,-1].astype(np.int)
                
        pathDataset = 'data/{}/{}'.format(ds,f)
        if not os.path.exists(pathDataset):
            os.makedirs(pathDataset)   


        #main approach: remove samples from majority class until balancing the dataset
        perform_under(us_obj=us1, output=output, X=X, y=Y, X_test=X_test, y_test=Y_test,
                      fold=f, ds=ds, majority_class=majority_class, minority_class=minority_class, 
                      k_max=k_max, exec_time=end_time)                     
        
        #1st variant: remove samples from majority class with negative scores  
        perform_under(us_obj=us2, output=output, X=X, y=Y, X_test=X_test, y_test=Y_test,
                      fold=f, ds=ds, majority_class=majority_class, minority_class=minority_class, 
                      k_max=k_max, exec_time=end_time)

       #2st variant: remove samples from majority class with negative or zero scores
        perform_under(us_obj=us3, output=output, X=X, y=Y, X_test=X_test, y_test=Y_test,
                      fold=f, ds=ds, majority_class=majority_class, minority_class=minority_class, 
                      k_max=k_max, exec_time=end_time)
