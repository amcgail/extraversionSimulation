# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 11:04:48 2017

@author: alec
"""
import numpy as np
from numpy.random import exponential
from numpy.linalg import inv

from os.path import join, dirname, exists
import os

basedir = join( dirname(__file__), ".." )
simdir = join( basedir, "simulations" )

def defaultMu(x):
    if x <= -1e2:
        return 0.5
        
    return abs( 1/(1 + np.exp(-x)) - 0.5 )

def renderCSV(rows, fn):
    with open(fn, 'w') as csvf:
        from csv import writer
        cw = writer(csvf)
        for r in rows:
            cw.writerow(r)

def getsimdir(name):
    d = join(simdir, name)
    if not exists(d):
        os.mkdir(d)
    return d

class simulation:
    p = 0.01
    eps1 = 0.01
    eps2 = 0.01
    eps3 = 0.01
    
    def __init__( self, alpha, mu=defaultMu, sigma=None, L=None ):
        self.initializeSnapshotMetrics()        
        
        self.t = 0
        self.alpha = alpha
        self.N = len(alpha)
        self.mu = mu
        
        self.L = L
        if self.L is None:
            self.L = ( np.zeros( (self.N, self.N) ) + 1 ) - np.identity(self.N)
        
        if sigma is None:
            self.sigma = np.zeros((self.N, self.N))
        else:        
            self.sigma = sigma
        
    def sigmaSum(self, i):
        return np.dot( self.sigma[i], self.L[i] )
    
    def s(self, i, sigma=None):
        if self.sigmaSum(i) == 0:
            if self.alpha[i] == 0:
                return 0
            
            return self.mu(1e10) #infinity
            
        return self.mu( 1 - self.alpha[i] / (self.sigmaSum(i)+1e-10) )
        
    def T(self, i):
        myS = self.s(i)
        return exponential( 1/(myS+1e-10) )
        
    def Tmin(self):
        mint = None
        mini = None
        for i in range(self.N):
            t = self.T(i)
            if mint is None or t < mint:
                mint = t
                mini = i
                
        return (mint, mini)
    
    def B(self):
        eigs = np.linalg.eigvals(self.sigma )
        eigs = np.real( eigs )
        #print("eigs", eigs, max(eigs))
        p = 1/(max( max(eigs)*2, 0.5 ))
        
        return inv( np.identity(self.N) - p * self.sigma )
        
    def breakUpWithSomeone(self, i):
        B = self.B()
        myB = B[i]
        
        probs = (self.sigma[i] * self.L[i] > 0) * (1 / (myB+1e-10) + self.eps1)
        #print("0",probs)
        #probs = np.multiply(probs, np.array([i!=j for j in range(self.N)]))
        probs = np.nan_to_num(probs)
        #print("1",probs)
        probs = probs / sum(probs)
        #print("2",probs)
        
        unluckyFellow = np.random.choice(range(self.N), p=probs)
        uF = unluckyFellow
        
        self.sigma[i,uF] = 0
        self.sigma[uF,i] = 0
        #print("%s broke up with %s" % (i, uF)
        
    def findNewFriend(self, i):
        B = self.B()
        myB = B[i]
        probs1 = np.array([ 
            (self.sigmaSum(j) < self.alpha[j]) * self.s(j) 
            for j in range(self.N) 
        ]) + self.eps2
        
        probs1 = np.multiply(probs1, np.array([i!=j for j in range(self.N)]))
        probs1 = probs1 / sum(probs1)
        
        probs2 = myB + self.eps3
        probs2 = probs2 / sum(probs2)
        
        probs = np.multiply( probs1, probs2 )
        probs = probs / sum(probs)
        
        luckyFellow = np.random.choice(range(self.N), p=probs)
        lF = luckyFellow
        
        aNeed = max( self.alpha[i] - self.sigmaSum(i), 0 )
        bNeed = max( self.alpha[lF] - self.sigmaSum(lF), 0 )
        timeToSpend = np.random.uniform( aNeed , bNeed )
        
        self.sigma[i,lF] += timeToSpend
        self.sigma[lF, i] += timeToSpend
        
        #print("%s hooked up with %s" % (i, lF)
        
    def takeOpportunity(self, i):
        s = self.sigmaSum(i)
        if s > self.alpha[i]:
            self.breakUpWithSomeone(i)
        elif s < self.alpha[i]:
            self.findNewFriend(i)
            
    def initializeSnapshotMetrics(self):
        self.sigmaLog = []
        self.alphaLog = {}
        self.LLog = {}        
        
        self.degreeDistLog = []
        
    def snapShotMetrics(self):
        from collections import Counter
        
        degreeDist = dict( Counter( [ sum(self.sigma[i] > 0) for i in range(self.N) ] ) )
        self.degreeDistLog.append( (self.t, degreeDist) )
        
        self.sigmaLog.append( (self.t, np.copy(self.sigma)) )
        self.alphaLog[self.t] = np.copy(self.alpha)
        self.LLog[self.t] = np.copy(self.L)
    
    def oneRound(self):
        mint, mini = self.Tmin()       
        
        self.t += mint
        #print("%s is choosing..." % mini)
        self.takeOpportunity(mini)
        self.snapShotMetrics()
        
    def runSimulation(self, nsteps=None, time=None):
        if nsteps is None and time is None:
            nsteps = 100
        
        from time import sleep
        
        if time is not None:
            endTime = self.t + time
        
        printStep = 50
        
        i = 0
        while 1:
            i += 1
            if i % printStep == 0:
                if nsteps is not None:
                    print("Step %s / %s" % (i, nsteps))
                    print(" -- t=%.2f" % (self.t))
                if time is not None:
                    print("Step %s" % (i))
                    print(" -- t=%.2f / %.2f" % (self.t, endTime))
                    
                if printStep*10 < i:
                    printStep *= 1.5
            
            self.oneRound()
            
            if time is not None and self.t > endTime:
                break
            if nsteps is not None and i > nsteps:
                break
            
    def addPerson(self, alph):
        # add their alpha...
        self.alpha = np.append( self.alpha, alph )
        # they have no interactions yet
        self.sigma = np.append( self.sigma, np.zeros((1,self.N)), axis=0 )
        self.sigma = np.append( self.sigma, np.zeros((self.N+1,1)), axis=1 )
        
        self.N += 1
        
        # return index of new person
        print("Person with extroversion %0.2f added." % alph)
        return self.N - 1
    
    def killPerson(self, i):
        self.sigma = np.delete( self.sigma, i, 0 )
        self.sigma = np.delete( self.sigma, i, 1 )
        
        self.L = np.delete( self.L, i, 0 )
        self.L = np.delete( self.L, i, 1 )
        
        self.alpha = np.delete( self.alpha, i )
        self.N -= 1
        
        print("Person %d killed" % i)

    def randomDisaster(self, npeople):
        assert npeople <= self.N
        
        from random import choice
        for i in range(npeople):
            self.killPerson( choice( range( self.N ) ) )


    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # ++++++++++++++++    export everything    ++++++++++++++++++++++++
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


    def exportGraphCSV(self, fn):
        from csv import writer
        exportdir = getsimdir(fn)
        nodeFn = join(exportdir, "nodes.csv")
        edgeFn = join(exportdir, "edges.csv")

        with open(nodeFn, 'w') as csvf:
            cw = writer(csvf)

        with open(edgeFn, 'w') as csvf:
            cw = writer(csvf)
            h = ["StartTime", "EndTime", "Source", "Target", "weight"]
            cw.writerow(h)

            all_t = [ t for t,_ in self.sigmaLog ]
            next_t = {
                all_t[i]: all_t[i+1]
                for i in range(len(all_t) - 1)
            }

            for t, sigma in self.sigmaLog[:-1]:
                N, _ = np.shape(self.sigma)

                for ni in range(N):
                    for nj in range(N):

                        friendship = sigma[ni,nj]
                        if friendship == 0:
                            continue

                        cw.writerow([ t, next_t[t], ni, nj, friendship ])

        print("Edges exported to %s" % edgeFn)
        print("Nodes exported to %s" % nodeFn)

    def exportDegrees(self, fn):    
        fn = join( getsimdir(fn), "deg.all.csv" )

        rows = []
        rows.append(["t", "degree","alpha","personi"])
        for t, sigma in self.sigmaLog:
            N,_ = np.shape(sigma)
            
            for i in range(N):
                deg = sum( sigma[i] > 0 )
                rows.append([ t, deg, self.alphaLog[t][i], i ])
        
        renderCSV( rows, fn )
        print("All degrees for all users exported to %s" % fn )
    def exportDegreeDist(self, fn):
        fn = join(getsimdir(fn), "deg.csv")

        rows = []
        rows.append(["t", "degree","freq"])
        for t, dist in self.degreeDistLog:
            for degree,freq in dist.items():
                rows.append([t,degree,freq])
        
        renderCSV( rows, fn )
        print("Degree distribution exported to %s" % fn)
    def exportUnhappinessEvolution(self, fn):
        fn = join(getsimdir(fn), "unhap.csv")

        rows = []
        rows.append(["t", "avgUnhap", "minUnhap", "maxUnhap", "stdUnhap"])
        for t, sigma in self.sigmaLog:
            N,_ = np.shape(sigma)
            unhap = [ self.mu( 1 - self.alphaLog[t][i] / ( np.dot( sigma[i], self.LLog[t][i] )+1e-10) ) for i in range(N) ]
            rows.append([
                t, np.mean(unhap), np.min(unhap), np.max(unhap), np.std(unhap)
            ])
        
        renderCSV( rows, fn )
        print("Unhappiness exported to %s" % fn)
    
    def exportAlphaSimilarity(self, fn):
        fn = join(getsimdir(fn), "alpha.csv")
        
        rows = []
        rows.append(["t", "personi", "alpha", "friendWeightedAvgAlpha", "friendAvgAlpha", "friendStdAlpha"])
        # this weights each person by how much time they are spent with... whatever
        for t, sigma in self.sigmaLog:
            N,_ = np.shape(sigma)
            for i in range(N):
                friendAlphas = [ self.alpha[j] for j in range(N) if sigma[i,j] > 0 ]
                if len(friendAlphas) == 0:
                    rows.append( [
                        t, i,
                        self.alphaLog[t][i], 
                        -1,
                        -1,
                        -1
                    ] )
                else:
                    rows.append( [
                        t, i,
                        self.alphaLog[t][i], 
                        np.average( self.alpha, weights = np.multiply( sigma[i], self.LLog[t][i] ) ),
                        np.average( friendAlphas ),
                        np.std( friendAlphas )
                    ] )
        print("Alpha similarity exported to %s" % fn)
                
        renderCSV( rows, fn )

    def exportEverything(self, fn):
        self.exportGraphCSV(fn)
        self.exportDegreeDist(fn)
        self.exportDegrees(fn)
        self.exportUnhappinessEvolution(fn)
        # self.exportAlphaSimilarity(fn)