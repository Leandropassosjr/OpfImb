# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 20:00:32 2020

@author: DANILO
"""

import sys
sys.path.append('..')

from hybrid.us_os import HYBRID
from undersampling.opf_us3 import OpfUS3

class US3O2PF(HYBRID):

	def variant(self):
		opf_us = OpfUS3()
		return opf_us
