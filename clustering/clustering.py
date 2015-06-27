from __future__ import division
from numpy import *
import numpy as np
import logging

best = None
bestDist = 2**255
bestClusters = None

# linear time, saves cost, penalizes away from the mode (as in paul's rbcr/
# unlike other clustering algos)
# allowing us to maintain incentive compatibility w/ rbcr game theory
# yet penalize people more accurately & precisely than pca or any other form of clustering
class clusternode:
    def __init__(self,vec,numItems=0,meanVec=None,rep=0,repVec=None, reporterIndexVec=None, dist=-1):
        # num of events would be == len(vec[i])
        self.vec=vec
        #numitems is num reporters in this cluster
        self.numItems=numItems
        self.meanVec=meanVec
        self.rep=rep
        self.repVec=repVec
        self.reporterIndexVec = reporterIndexVec
        self.dist = dist

def L2dist(v1,v2):
    return sqrt(sum((v1-v2)**2))

def newMean(cmax):
    weighted = zeros([cmax.numItems, len(cmax.vec[0])]).astype(float)
    for i in range(cmax.numItems):
        weighted[i,:] = cmax.vec[i]*cmax.repVec[i]
    x = sum(weighted, axis=0)
    mean = [y / cmax.rep for y in x]
    return(mean)

# expects a numpy array for reports and rep vector
def cluster(features, rep, times=1, threshold=0.50, distance=L2dist):
    #cluster the rows of the "features" matrix
    distances={}
    threshold = np.log10(len(features[0]))/1.7
    currentclustid=-1
    clusters = empty(1, dtype=object)
    for n in range(len(rep)):
        if(rep[n]==0.0):
            rep[n] = 0.00001
    for i in range(len(features)):
        # cmax is most similar cluster
        cmax = None
        shortestDist = 2**255
        for n in range(len(clusters)):
            if(n!=0):
                dist = L2dist(features[i], clusters[n].meanVec)
                if dist<shortestDist:
                    cmax = clusters[n]
                    shortestDist = dist
        if(cmax!=None and L2dist(features[i], cmax.meanVec) < threshold):
            cmax.vec = concatenate((cmax.vec, array([features[i]])))
            cmax.numItems += 1
            cmax.rep += rep[i]
            cmax.repVec = append(cmax.repVec, rep[i])
            cmax.meanVec = array(newMean(cmax))
            cmax.reporterIndexVec += [i]
        else:
            clusters = append(clusters, clusternode(array([features[i]]), 1, features[i], rep[i], array(rep[i]), [i]))
    clusters = delete(clusters, 0)
    clusters = process(clusters, len(features), times, features, rep, threshold)
    return clusters

def process(clusters, numReporters, times, features, rep, threshold):
    mode = None
    numInMode = 0
    global best
    global bestClusters
    global bestDist
    for i in range(len(clusters)):
        if(clusters[i].rep > numInMode):
            numInMode = clusters[i].rep
            mode = clusters[i]

    outcomes = np.ma.average(features, axis=0, weights=rep)

    # detect how far the "truthers" are away from actual outcomes
    # then choose closer mode as final truth cluster
    if(L2dist(mode.meanVec, outcomes)<bestDist):
        bestDist = L2dist(mode.meanVec, outcomes)
        best = mode
        bestClusters = clusters
    if(L2dist(mode.meanVec,outcomes)>1.07 and times==1):
        possAltCluster = cluster(features,rep,2,threshold*3)
        return(possAltCluster)


    for x in range(len(bestClusters)):
        bestClusters[x].dist = L2dist(best.meanVec,bestClusters[x].meanVec)

    distMatrix = zeros([numReporters, 1]).astype(float)
    for x in range(len(bestClusters)):
        for i in range(bestClusters[x].numItems):
            distMatrix[bestClusters[x].reporterIndexVec[i]] = bestClusters[x].dist
    repVector = zeros([numReporters, 1]).astype(float)
    for x in range(len(distMatrix)):
        repVector[x] = 1 - distMatrix[x]/(amax(distMatrix)+0.00000001)
    logging.warning(normalize(repVector))
    return(normalize(repVector))

def normalize(v):
    """Proportional distance from zero."""
    v = abs(v)
    if sum(v) == 0:
        v += 1
    return v / sum(v)


# the issue of a 0 report cluster: should not count it as truth if it is biggest cluster
# for now let's just pick the first cluster as mode one if same amount of weight as another cluster
# ban part filled ballots

# Notes:
    # normalized rep functions:
        # 1 / (1 + d[i]) ^ 2 - my favorite, penalizes liars quadratically away from the truth
        # 1 - x/maximum(x) - penalizes liars so hard, almost too much imo
        # 1/(value + epsilon) - least favorite

#TODO: 
    # categorical markets
    # scalar markets (scaling to 1 might still be a good idea) (actually do this before submit ballot)
    # what if two modes (e.g. _exact same # of rep on both, so 2 equal sized clusters w/ same magnitude)
    # perhaps just push consensus back, oh nvm this wouldn't be an issue, 65% threshold prevents it
    # from being decided (but rbcr is still done)
    # i suppose you could just in that case check which is more similar to the _rest_ of the clusters
        # (weighted of course) on _average_ and pick that one as the mode cluster
        # for now let's just pick the first cluster as mode one
    # or reweight to weighted avg then penalize particiption

#Clustering notes:
    #DBSCAN, ISODATA (like k means but don't need to pick k), DENCLUE (> DBSCAN?) (dbscan seems promising)
    #https://en.wikipedia.org/wiki/OPTICS_algorithm (similar to dbscan)