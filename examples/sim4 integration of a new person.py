# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 11:04:48 2017

@author: alec
"""
import numpy as np
from numpy.random import exponential
from numpy.linalg import inv

class simulation():
    p = 0.01
    eps1 = 0.01
    eps2 = 0.01
    eps3 = 0.01
    
    def __init__( self, alpha, mu, sigma=None ):
        self.initializeSnapshotMetrics()        
        
        self.t = 0
        self.alpha = alpha
        self.N = len(alpha)
        self.mu = mu
        
        if sigma is None:
            self.sigma = np.zeros((self.N, self.N))
        else:        
            self.sigma = sigma
        
    def sigmaSum(self, i):
        return sum( self.sigma[i] )
    
    def s(self, i):
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
        eigs = np.linalg.eigvals(self.sigma)
        eigs = np.real( eigs )
        #print("eigs", eigs, max(eigs))
        p = 1/(max( max(eigs)*2, 0.5 ))
        
        return inv( np.identity(self.N) - p * self.sigma )
        
    def breakUpWithSomeone(self, i):
        B = self.B()
        myB = B[i]
        
        probs = (self.sigma[i] > 0) * (1 / (myB+1e-10) + self.eps1)
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
        #print "%s broke up with %s" % (i, uF)
        
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
        
        #print "%s hooked up with %s" % (i, lF)
        
    def takeOpportunity(self, i):
        s = self.sigmaSum(i)
        if s > self.alpha[i]:
            self.breakUpWithSomeone(i)
        elif s < self.alpha[i]:
            self.findNewFriend(i)
            
    def initializeSnapshotMetrics(self):
        self.disLog = []
        self.avgDisLog = []
        self.stdDisLog = []
        
        self.sigmaLog = []
        
    def snapShotMetrics(self):
        dis = [ self.alpha[i] - self.sigmaSum(i) for i in range(self.N) ]
        self.disLog.append( (self.t, dis) )
        
        self.sigmaLog.append( (self.t, np.copy(self.sigma)) )
    
    def oneRound(self):
        mint, mini = self.Tmin()       
        
        self.t += mint
        #print "%s is choosing..." % mini
        self.takeOpportunity(mini)
        self.snapShotMetrics()
        
    def runSimulation(self, nsteps=None, ntime=None):
        if nsteps is None and ntime is None:
            nsteps = 100
        
        from time import sleep
        i = 0
        while 1:
            i += 1
            if i % 10 == 0:
                print "Step %s" % i
                print "-- t=%s" % self.t
            
            self.oneRound()
            
            if ntime is not None and self.t > ntime:
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
        return self.N - 1
    
    def exportGraphCSV(self, fn):
        from csv import writer
        nodeFn = "%s.nodes.csv" % fn
        edgeFn = "%s.edges.csv" % fn
        
        with open(nodeFn, 'wb') as csvf:
            cw = writer(csvf)
            
            
        with open(edgeFn, 'wb') as csvf:
            cw = writer(csvf)
            h = ["StartTime", "EndTime", "Source", "Target", "weight"]
            cw.writerow(h)
            
            for ni in range(self.N):
                for nj in range(self.N):
                    lastT = None
                    lastS = None
                    for t, sigma in self.sigmaLog:
                        shp = np.shape( sigma )
                        if ni >= shp[0] or nj >= shp[1]:
                            continue

                        if lastT is not None:
                            cw.writerow([ lastT, t, ni, nj, lastS ])
                        
                        if sigma[ni,nj] == 0:
                            lastT = None
                            lastS = None
                        else:
                            lastT = t
                            lastS = sigma[ni,nj]
        
def mu(x):
    return abs( 1/(1 + np.exp(-x)) - 0.5 )
    
s = simulation( np.arange(1, 10, 0.1), mu )
s.runSimulation( ntime=15 )

s.addPerson(15)

s.runSimulation( ntime=20 )

s.exportGraphCSV( "addNewPeople" )