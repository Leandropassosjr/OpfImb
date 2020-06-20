import numpy as np
from undersampling.opf_us import OpfUS
from common.common import COMMON
import sys

if len(sys.argv) <= 1:
	print('Usage: %s <train dataset> <validation dataset (optional)>' % sys.argv[0])
	raise SystemExit

train = np.loadtxt(sys.argv[1],delimiter=',', dtype=np.float32)

valid = None
if len(sys.argv)>=3:
	valid = np.loadtxt(sys.argv[2],delimiter=',', dtype=np.float32)		
	concat = np.concatenate((train, valid))
	X = concat[:,:-1]
	y = concat[:,-1].astype(np.int) 
else:
	X = train[:,:-1]
	y = train[:,-1].astype(int)

opf_us = OpfUS()
X_res, y_res = opf_us.fit_resample(X, y, valid)

path = 'data'
common = COMMON()
common.saveDataset(X_res, y_res, path, opf_us.__class__.__name__)
