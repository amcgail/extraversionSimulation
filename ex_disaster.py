# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 19:34:30 2017

@author: alec
"""

from exSim import simulation
import numpy as np

s = simulation( np.arange( 1, 10, 0.1 ) )
s.runSimulation( time=5 )
s.randomDisaster( 25 )
s.runSimulation( time=5 )

s.exportGraphCSV( "out" )
s.exportDegreeDist( "out" )
s.exportDegrees( "out" )
s.exportUnhappinessEvolution( "out" )