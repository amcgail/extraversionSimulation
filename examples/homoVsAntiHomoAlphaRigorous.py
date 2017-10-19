# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 15:12:28 2017

@author: alec
"""

from exSim import simulation
import numpy as np

# 100 individuals, as usual
alpha = np.arange(0,10,0.1)
# distance in alpha indicates preference of both individuals
# rescaled such that maximum preference = 1
Lanti = np.array([[ abs(i-j) / max([ abs(i-j2) for j2 in alpha]) for j in alpha] for i in alpha])

# now we simply rearrange each row
Lhomo = np.copy( Lanti )
N,_ = np.shape(Lhomo)
for i in range(N):
    sortedLikes = sorted( list( Lhomo[i] ) )
    # note that we skip ourselves. we should stay zero...
    for j in sorted( range(len(alpha)), key=lambda x: abs(x-i) )[1:]:
        Lhomo[i,j] = sortedLikes.pop()

for i in range(50):

    sAnti = simulation(alpha, L=Lanti)
    sHomo = simulation(alpha, L=Lhomo)
    
    sAnti.runSimulation(ntime=500)
    sHomo.runSimulation(ntime=500)

    sHomo.exportAlphaSimilarity("antiHomophily/%s.homo.alpha.csv"%i)
    sHomo.exportDegreeDist("antiHomophily/%s.homo.degDist.csv"%i)
    sHomo.exportUnhappinessEvolution("antiHomophily/%s.homo.unhap.csv"%i)
    
    sAnti.exportAlphaSimilarity("antiHomophily/%s.anti.alpha.csv"%i)
    sAnti.exportDegreeDist("antiHomophily/%s.anti.degDist.csv"%i)
    sAnti.exportUnhappinessEvolution("antiHomophily/%s.anti.unhap.csv"%i)
