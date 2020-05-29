import pickle
import os
import cv2
from time import time
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from fingerUtil import preTreatment, createHistogram

pathToTrainDir = 'D:/ProgramData/programmes/PyCharm/dnn_data/fingerprints/training'
pathToSaveDir = ''
if pathToSaveDir != '':
    pathToSaveDir = pathToSaveDir + '/'
clustersNumber = 750
sift = cv2.xfeatures2d.SIFT_create()

def getAllPic(path):
    """get all pictures' names in subdirectories
    Intended architecture : several directories with no subdirectory

    Keyword arguments:
        path -- path to the directory which contains subdirectories"""

    picClasses=[]
    picPaths = []
    dirList = os.listdir(path)
    for directory in dirList:
        picList = os.listdir(path+'/'+directory)
        for pic in picList:
            picPaths.append(path+'/'+directory+'/'+pic)
            picClasses.append(directory)
    return picPaths, picClasses


def loadAllPic(path):
    """Load all pictures in designated directory
    Intended architecture : several directories with no subdirectory

    returns:
        sortedDescriptors -- a dictionnary with key: class of the picture
                                 value: descriptors
        allDescriptors -- A list of every descriptors

    Keyword arguments:
        path -- path to the directory which contains subdirectories"""

    sortedDescriptors = {}
    allDescriptors = []
    picPaths, picClasses = getAllPic(path)
    for i in range(len(picPaths)):
        picClass = picClasses[i]
        picture = preTreatment(picPaths[i])
        _, des = sift.detectAndCompute(picture, None)
        try:
            sortedDescriptors[picClass].append(des)
        except:
            sortedDescriptors[picClass] = [des]
        allDescriptors.extend(des)
    return sortedDescriptors, allDescriptors


def kmeans(clustersNumber, allDescriptors):
    """Find k clusters, which's centres will be our visual words

    returns:
        visualWords -- an array that contains central points of clusters

    Keyword arguments:
        allDescriptors -- A list of every descriptors"""

    kmeans = KMeans(n_clusters=clustersNumber, n_init=10)
    kmeans.fit(allDescriptors)
    visualWords = kmeans.cluster_centers_
    return visualWords


if __name__ == '__main__':

    start = time()
    print("pic loading")
    sortedRefDes, allRefDes = loadAllPic(pathToTrainDir)
    print("clustering, this will take a while...")
    visualWords = kmeans(clustersNumber, allRefDes)
    print("matcher creation")
    matcherRef = NearestNeighbors(n_neighbors=2, algorithm='auto', metric='euclidean').fit(visualWords)
    print("creating histo")
    refHist, refClasses = createHistogram(sortedRefDes, matcherRef, clustersNumber)
    print("matcher creation")
    matcher = NearestNeighbors(n_neighbors=2, algorithm='auto', metric='euclidean').fit(refHist)
    print("prep done.")
    end = time()

    print("Saving")
    matcherPickle = open('matcherPickle_file', 'wb')
    pickle.dump(matcher, matcherPickle)
    matcherRefPickle = open('matcherRefPickle_file', 'wb')
    pickle.dump(matcherRef, matcherRefPickle)
    with open('refClassesPickle_file.pickle', 'wb') as f:
        pickle.dump(refClasses, f, pickle.HIGHEST_PROTOCOL)
    print(refClasses)
    print("done.")
