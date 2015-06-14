from numpy import *

class clusternode:
    def __init__(self,vec,numItems=0,meanVec=None):
        self.vec=vec
        self.numItems=numItems
        self.meanVec=meanVec

def L2dist(v1,v2):
    return sqrt(sum((v1-v2)**2))
    
def L1dist(v1,v2):
    return sum(abs(v1-v2))

def newMean(cmax):
    x = sum(cmax.vec, axis=0)
    divisor = cmax.numItems
    mean = [y / divisor for y in x]
    return(mean)

def cluster(features,distance=L1dist):
    #cluster the rows of the "features" matrix
    distances={}
    currentclustid=-1
    clusters = []
    for i in range(len(features)):
        cmax = None
        shortestDist = 2**255
        for n in range(len(clusters)):
            dist = L2dist(array(features[i]), array(clusters[n].meanVec))
            if dist<shortestDist:
                cmax = clusters[n]
                shortestDist = dist
        if(cmax!=None and L2dist(array(features[i]), array(cmax.meanVec)) < 1.50):
            cmax.vec.append(features[i])
            cmax.numItems += 1
            cmax.meanVec = newMean(cmax)
        else:
            clusters.append(clusternode([features[i]], 1, features[i]))
    return clusters