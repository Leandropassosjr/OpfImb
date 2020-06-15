from undersampling.opf_us1 import OpfUS1
from undersampling.opf_us2 import OpfUS2
from undersampling.opf_us3 import OpfUS3
from undersampling.opf_us4 import OpfUS4
from undersampling.opf_us import OpfUS

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
    
    start_time = time()
    X_res, y_res = opf_us_obj.variant(output, X, y, majority_class, minority_class)
    end_time = time() - start_time + exec_time
    
    approach = opf_us_obj.__class__.__name__
    
    opf_us_obj.saveDataset(X_res, y_res, pathDataset, approach)
    opf_us_obj.saveResults(X_res, y_res, X_test, y_test, ds, f, approach, minority_class, end_time)


datasets = ['vertebral_column']
#paper uses 20 folds, so the next line runs as follows:
#folds = np.arange(1,21)
folds = np.arange(1,2)

# Objects for undersampling
opf_us1 = OpfUS1()
opf_us2 = OpfUS2()
opf_us3 = OpfUS3()
opf_us4 = OpfUS4()
opf_us = OpfUS()

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
        output, majority_class, minority_class = opf_us1.run(X, Y)
        end_time = time() -start_time
        
        X = X[:len(train),...]
        Y = Y[:len(train),...]
        output = output[:len(train),...]
        
        X_test = test[:,:-1]
        Y_test = test[:,-1].astype(np.int)
                
        pathDataset = 'data/{}/{}'.format(ds,f)
        if not os.path.exists(pathDataset):
            os.makedirs(pathDataset)   


        #main approach: remove samples from majoritary class until balancing the dataset
        perform_under(us_obj=opf_us, output=output, X=X, y=Y, X_test=X_test, y_test=Y_test,
                      fold=f, ds=ds, majority_class=majority_class, minority_class=minority_class, 
                      exec_time=end_time)                     
        
        #1st variant: remove samples from majoritary class with negative scores  
        perform_under(us_obj=opf_us1, output=output, X=X, y=Y, X_test=X_test, y_test=Y_test,
                      fold=f, ds=ds, majority_class=majority_class, minority_class=minority_class, 
                      exec_time=end_time)

       #2st variant: remove samples from majoritary class with negative or zero scores
        perform_under(us_obj=opf_us2, output=output, X=X, y=Y, X_test=X_test, y_test=Y_test,
                      fold=f, ds=ds, majority_class=majority_class, minority_class=minority_class, 
                      exec_time=end_time)
        
        #3st variant: remove all samples with negative
        perform_under(us_obj=opf_us3, output=output, X=X, y=Y, X_test=X_test, y_test=Y_test,
                      fold=f, ds=ds, majority_class=majority_class, minority_class=minority_class, 
                      exec_time=end_time)
        
        #4st variant: remove samples from majoritary class with negative or zero scores 
        perform_under(us_obj=opf_us4, output=output, X=X, y=Y, X_test=X_test, y_test=Y_test,
                      fold=f, ds=ds, majority_class=majority_class, minority_class=minority_class, 
                      exec_time=end_time)        
        
      
        
        
