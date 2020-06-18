from __future__ import absolute_import
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, f1_score
import numpy as np
import os
import abc
import sys

sys.path.append('..')
from opf.models.supervised import SupervisedOPF

class US(metaclass=abc.ABCMeta):

    def __init__(self):
        self.opfSup = SupervisedOPF(distance='log_squared_euclidean', pre_computed_distance=None)

    def __classify(self, x_train,y_train, x_valid, y_valid, minority_class):
        # Training the OPF                
        indexes = np.arange(len(x_train))
        self.opfSup.fit(x_train, y_train,indexes)

        # Prediction of the validation samples
        y_pred,_ = self.opfSup.predict(x_valid)
        y_pred = np.array(y_pred)
        
        # Validation measures for this k nearest neighbors
        accuracy = accuracy_score(y_valid, y_pred)
        recall = recall_score(y_valid, y_pred, pos_label=minority_class) # assuming that 2 is the minority class
        f1 = f1_score(y_valid, y_pred, pos_label=minority_class)
        return accuracy, recall, f1, y_pred

    def __computeScore(self, labels, preds, conqs, score):
        
        for i in range(len(labels)):
            if labels[i]==preds[i]:
                score[conqs[i]]+=1
            else:
                score[conqs[i]]-=1

    def __runOPF(self, X_train,y_train,index_train,X_test,y_test,index_test, score):
        # Creates a SupervisedOPF instance
        opf = SupervisedOPF(distance='log_squared_euclidean',
                            pre_computed_distance=None)

        # Fits training data into the classifier
        opf.fit(X_train, y_train, index_train)
        
        # Predicts new data
        preds, conqs = opf.predict(X_test)
        
        self.__computeScore(y_test, preds, conqs, score)
    
    def run(self, X, Y):
        indices = np.arange(len(X))
        
        # Counting the number of samples in each class
        (uniques, frequency) = np.unique(Y, return_counts=True)
        
        # Indices of the minority and majority classes
        idx_min = np.argmin(frequency)
        idx_max = np.argmax(frequency)
        
        # Number of samples 
        minority_class, majority_class = uniques[idx_min], uniques[idx_max]        
        
        # Create stratified k-fold subsets
        kfold = 5 # no. of folds
        skf = StratifiedKFold(kfold, shuffle=True,random_state=1)
        skfind = [None] * kfold  # skfind[i][0] -> train indices, skfind[i][1] -> test indices
        cnt = 0
        for index in skf.split(X, Y):
            skfind[cnt] = index
            cnt += 1        
        
        score = np.zeros((5,len(X)))

        for i in range(kfold):
            train_indices = skfind[i][0]   
            test_indices = skfind[i][1]
            X_train = X[train_indices]
            y_train = Y[train_indices]
            index_train = indices[train_indices]
        
        
            X_test = X[test_indices]
            y_test = Y[test_indices]
            index_test = indices[test_indices]
            self.__runOPF(X_train,y_train,index_train,X_test,y_test,index_test, score[i])
        

        output=  np.zeros((len(indices),8))

        score_t = np.transpose(score)
        output[:,0] =indices
        output[:,1] =Y
        output[:,2] =np.sum(score_t,axis=1)
        output[:,3:] =score_t

        return output, majority_class, minority_class
    
    def saveResults(self, X_train,Y_train, X_test, Y_test,  ds,f, approach, minority_class, exec_time, path_output):

        path = '{}/down_{}/{}/{}'.format(path_output,approach,ds,f)
        if not os.path.exists(path):
            os.makedirs(path)

        results_print=[]
        accuracy, recall, f1, pred = self.__classify(X_train,Y_train, X_test, Y_test, minority_class)
        results_print.append([0,accuracy, recall, f1, exec_time])

        np.savetxt('{}/pred.txt'.format(path), pred, fmt='%d')
        np.savetxt('{}/results.txt'.format(path), results_print, fmt='%d,%.5f,%.5f,%.5f,%.5f')

    def saveDataset(self, X_train,Y_train, pathDataset,approach):
        DS = np.insert(X_train,len(X_train[0]),Y_train , axis=1)
        np.savetxt('{}/train_{}.txt'.format(pathDataset, approach),DS,  fmt='%.5f,'*(len(X_train[0]))+'%d')    
    
    @abc.abstractmethod
    def variant(self, output, X, Y,  majority_class, minority_class):
        return
    
    @abc.abstractmethod
    def fit_resample(self, X, y):
        return
