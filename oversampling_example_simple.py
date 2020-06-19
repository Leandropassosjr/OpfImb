import numpy as np
from oversampling.o2pf import O2PF
from common.common import COMMON
import sys

if len(sys.argv) <= 2:
	print('Usage: %s <train dataset>  <k_max>' % sys.argv[0])
	raise SystemExit

train = np.loadtxt(sys.argv[1],delimiter=',', dtype=np.float32)

X = train[:,:-1]
y = train[:,-1].astype(int)

o2pf = O2PF()
X_res, y_res = o2pf.fit_resample( X, y, int(sys.argv[2]))

path = 'data'
common = COMMON()
common.saveDataset(X_res, y_res, path, o2pf.__class__.__name__)
