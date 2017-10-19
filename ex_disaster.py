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

"""
Step 50
 -- t=2.53 / 25.00
Step 100
 -- t=6.31 / 25.00
Step 150
 -- t=15.84 / 25.00
Person 28 killed
Person 70 killed
Person 21 killed
Person 72 killed
Person 6 killed
Person 47 killed
Person 51 killed
Person 29 killed
Person 55 killed
Person 63 killed
Person 0 killed
Person 73 killed
Person 3 killed
Person 74 killed
Person 12 killed
Person 46 killed
Person 66 killed
Person 29 killed
Person 10 killed
Person 4 killed
Person 55 killed
Person 52 killed
Person 8 killed
Person 38 killed
Person 29 killed
Step 50
 -- t=35.32 / 50.02
Step 100
 -- t=49.94 / 50.02
Edges exported to out.edges.csv
Nodes exported to out.nodes.csv
Degree distribution exported to out.deg.csv
All degrees for all users exported to out.deg.all.csv
Unhappiness exported to out..unhap.csv
"""
