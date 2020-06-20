import numpy as np
from hybrid.us1_o2pf import US1O2PF
from common.common import COMMON
import sys

if len(sys.argv) <= 2:
	print('Usage: %s <train dataset> <k_max> <validation dataset (optional)>' % sys.argv[0])
	raise SystemExit

train = np.loadtxt(sys.argv[1],delimiter=',', dtype=np.float32)

valid = None
if len(sys.argv)>=4:
	valid = np.loadtxt(sys.argv[3],delimiter=',', dtype=np.float32)		
	concat = np.concatenate((train, valid))
	X = concat[:,:-1]
	y = concat[:,-1].astype(np.int) 
else:
	X = train[:,:-1]
	y = train[:,-1].astype(int)

hybrid_obj = US1O2PF(k_max=int(sys.argv[2]))

X_res, y_res = hybrid_obj.fit_resample( X, y, valid)

path = 'data'
common = COMMON()
common.saveDataset(X_res, y_res, path, hybrid_obj.__class__.__name__)
