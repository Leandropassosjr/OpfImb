# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 19:43:35 2020

@author: DANILO
"""

import sys
sys.path.append('..')

from hybrid.us_os import HYBRID
from undersampling.opf_us1 import OpfUS1

class US1O2PF(HYBRID):

	def variant(self):
		opf_us = OpfUS1()
		return opf_us
