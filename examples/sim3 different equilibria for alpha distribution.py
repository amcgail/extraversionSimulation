# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 11:04:48 2017

@author: alec
"""
import numpy as np
from numpy.random import exponential
from numpy.linalg import inv

from exSim import simulation
from exSim import renderCSV
    
if True:
    finalVals = []
    finalVals += [["spread", "sSum", "sInvSumInv", "sSumStd", "sInvSumInvStd"]]
    for i in np.arange(0.5, 9.5, 0.5):
        thisSample = []
        for j in range(10):
            print "Running simulation %s,%s" % (i,j)
            s = simulation( np.maximum( np.arange(10, 10-i, -i/50), 0 ) )
            s.runSimulation( ntime=50 )
            
            svals = np.array([ s.s(j) for j in range(s.N) ])
            thisSample.append( [np.sum(svals), 1/np.sum(1/svals) ] )
            
        finalVals.append( [
            i, 
            np.average([x[0] for x in thisSample]), 
            np.average([x[1] for x in thisSample]), 
            np.std([x[0] for x in thisSample]), 
            np.std([x[1] for x in thisSample])
        ] )
        
    renderCSV(finalVals, "alphaSpreadAffectsUtility.csv")

if False:
    finalVals = []
    finalVals += [["spread", "sSum", "sInvSumInv"]]
    i = 10
    
    for j in range(500):
        print "Running simulation %s,%s" % (i,j)
        s = simulation( np.maximum( np.arange(10, 10-i, -i/50), 0 ) )
        s.runSimulation( ntime=200 )
        
        svals = np.array([ s.s(j) for j in range(s.N) ])
        finalVals.append( [i, np.sum(svals), 1/np.sum(1/svals) ] )
        
    renderCSV(finalVals, "utilitySpreadForGivenAlpha.csv")