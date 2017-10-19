# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 13:42:38 2017

@author: alec
"""

from exSim import simulation
import numpy as np

alphas = np.arange(0,10,0.1)
L = np.array( [ [ abs(i-j) *(j!=i) for j in alphas ] for i in alphas ] )
s = simulation(alphas, L=L)

s.runSimulation(ntime=500)
