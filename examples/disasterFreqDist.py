# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 19:34:30 2017

@author: alec
"""

from exSim import simulation
import numpy as np

#s = simulation( np.maximum( 0.1, np.random.normal(20, 10, 100) ) )
s = simulation( np.concatenate( [ np.maximum( 0, np.random.normal(5, 2, 25) ), np.maximum(0, np.random.normal(10, 2, 25)) ] ) )
s.runSimulation( ntime=100 )
s.randomDisaster( 25 )
s.runSimulation( ntime=200 )

s.exportDegreeDist( "normalLargeVarianceWithDisaster.deg.csv" )

np.concatenate( [np.random.normal(20, 10, 2), np.random.normal(20, 10, 2)] )