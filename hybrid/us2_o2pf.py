# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 19:59:22 2020

@author: DANILO
"""

import sys
sys.path.append('..')

from hybrid.us_os import HYBRID
from undersampling.opf_us2 import OpfUS2

class US2O2PF(HYBRID):

	def variant(self):
		opf_us = OpfUS2()
		return opf_us
