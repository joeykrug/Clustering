from __future__ import division
from numpy import *
import logging

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
    logging.warning(mean)
    return(mean)

# expects a numpy array for reports and rep vector
def cluster(features, rep, distance=L2dist):
    #cluster the rows of the "features" matrix
    distances={}
    currentclustid=-1
    clusters = []
    for n in range(len(rep)):
        if(rep[n]==0.0):
            rep[n] = 0.00001
    for i in range(len(features)):
        # cmax is most similar cluster
        cmax = None
        shortestDist = 2**255
        for n in range(len(clusters)):
            dist = L2dist(features[i], clusters[n].meanVec)
            if dist<shortestDist:
                cmax = clusters[n]
                shortestDist = dist
        if(cmax!=None and L2dist(features[i], cmax.meanVec) < 0.50):
            cmax.vec = concatenate((cmax.vec, array([features[i]])))
            cmax.numItems += 1
            cmax.rep += rep[i]
            cmax.repVec = append(cmax.repVec, rep[i])
            cmax.meanVec = array(newMean(cmax))
            cmax.reporterIndexVec += [i]
        else:
            clusters.append(clusternode(array([features[i]]), 1, features[i], rep[i], array(rep[i]), [i]))
    clusters = process(clusters, len(features))
    return clusters

def process(clusters, numReporters):
    mode = None
    numInMode = 0
    for i in range(len(clusters)):
        if(clusters[i].rep > numInMode):
            numInMode = clusters[i].rep
            mode = clusters[i]
    for x in range(len(clusters)):
        clusters[x].dist = L2dist(mode.meanVec,clusters[x].meanVec)

    distMatrix = zeros([numReporters, 1]).astype(float)
    for x in range(len(clusters)):
        for i in range(clusters[x].numItems):
            distMatrix[clusters[x].reporterIndexVec[i]] = clusters[x].dist
    repVector = zeros([numReporters, 1]).astype(float)
    for x in range(len(distMatrix)):
        repVector[x] = 1 - distMatrix[x]/amax(distMatrix)
    return(normalize(repVector))

def normalize(v):
    """Proportional distance from zero."""
    v = abs(v)
    if sum(v) == 0:
        v += 1
    return v / sum(v)

# then normalize them per below
# the issue of a 0 report cluster: should not count it as truth if it is biggest cluster
# for now let's just pick the first cluster as mode one if same amount of weight as another cluster

# meeds a 2nd pass clustering --- if dist are similar, should be a new cluster


# Notes on Outcomes/Rbcr
    # for rbcr presumably normalize my distance metrics then use that in the smoothing

    # normalized rep functions:
        # 1 / (1 + d[i]) ^ 2 - my favorite, penalizes liars quadratically away from the truth
        # 1 - x/maximum(x) - penalizes liars so hard, almost too much imo
        # 1/(value + epsilon) - least favorite

    # outcome is easy, weighted avg of report matrix w/ weighted avg for zeroes


#TODO: 
    # categorical markets
    # scalar markets
    # clustering vs take mode, make that truth, then penalize everyone individually based on dist from mode
        # or is it cheaper to cluster, then penalize the clusters (e.g. not have to take time to do all individually)
    # what if two modes (e.g. _exact same # of rep on both, so 2 equal sized clusters w/ same magnitude)
    # perhaps just push consensus back, oh nvm this wouldn't be an issue, 65% threshold prevents it
    # from being decided (but rbcr is still done)
    # i suppose you could just in that case check which is more similar to the _rest_ of the clusters
        # (weighted of course) on _average_ and pick that one as the mode cluster
        # for now let's just pick the first cluster as mode one
    # distance threshold based on numevents and scalars perhaps log(numEvents)*2
    # don't allow half-full ballots,etc.
    # or reweight to weighted avg then penalize particiption

#Clustering notes:
    # There exists a modification to BSAS called modified BSAS (MBSAS), which runs
    #twice through the samples. It overcomes the drawback that a final cluster for a single
    #sample is decided before all the clusters have been created. The first phase of the
    #algorithm creates the clusters (just like 2b in BSAS) and assigns only a single sample to
    #each cluster. Then the second phase runs through the remaining samples and classifies
    #them to the created clusters (step 2c in BSAS). 

    #DBSCAN, ISODATA (like k means but don't need to pick k), DENCLUE (> DBSCAN?) (dbscan seems promising)
    #https://en.wikipedia.org/wiki/OPTICS_algorithm (similar to dbscan)