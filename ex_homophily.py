# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 15:12:28 2017

@author: alec
"""

from exSim import simulation
import numpy as np

# 100 individuals, as usual
alpha = np.arange(0,10,0.1)

# The following code creates antihomophily and homophily based on extraversion

# ANTI: distance in alpha indicates preference of both individuals
# rescaled such that maximum preference = 1
Lanti = np.array([[ abs(i-j) / max([ abs(i-j2) for j2 in alpha]) for j in alpha] for i in alpha])

# HOMO:
# now we simply rearrange each row
Lhomo = np.copy( Lanti )

N,_ = np.shape(Lhomo)

for i in range(N):
    sortedLikes = sorted( list( Lhomo[i] ) )
    # note that we skip ourselves. we should stay zero...
    for j in sorted( range(len(alpha)), key=lambda x: abs(x-i) )[1:]:
        Lhomo[i,j] = sortedLikes.pop()

# AND SIMULATE

for i in range(50):
    print "Running simulation %s/50" % i

    sAnti = simulation(alpha, L=Lanti)
    sHomo = simulation(alpha, L=Lhomo)
    
    sHomo.runSimulation(ntime=500)

    sHomo.exportAlphaSimilarity("%s.homo"%i)
    sHomo.exportDegreeDist("%s.homo"%i)
    sHomo.exportUnhappinessEvolution("antiHomophily/%s.homo"%i)

    sAnti.runSimulation(ntime=500)
    
    sAnti.exportAlphaSimilarity("%s.anti"%i)
    sAnti.exportDegreeDist("%s.anti"%i)
    sAnti.exportUnhappinessEvolution("%s.anti"%i)
