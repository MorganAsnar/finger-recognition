#INSPIRED BY https://medium.com/@aybukeyalcinerr/bag-of-visual-words-bovw-db9500331b2f

#TROUVER UN MOYEN POUR NE PAS MATCH SI LE VOISIN EST TROP LOIN LIGNE 150 SI LES DEUX PLUS PROCHES VOISINS ONT UN RAPPORT DE DISTANCE TROP PRES DE 1, ON DISCARD LE DESCRIPTEUR?

import numpy as np
import cv2
import os
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from time import time
import pandas as pd
import matplotlib.pyplot as plt

pathToDir = 'D:/ProgramData/programmes/PyCharm/dnn_data/fingerprints'
pathToTrainDir = pathToDir + '/training'
pathToValDir = pathToDir + '/validation'
resolution = (300,400)
rotated_resolution = (400,300)
clustersNumber = 750
acceptableConfidence = 0.75
acceptableDist = 1

sift = cv2.xfeatures2d.SIFT_create()

def deginrad(degree):
    """Take a degree angle, return the gradiant value"""
    radiant = 2*np.pi/360 * degree
    return radiant

#Gabor filter parameter
#Mat cv::getGaborKernel	(Size ksize,double sigma,double theta,double lambd,double gamma,double psi = CV_PI *0.5,int ktype = CV_64F)
#https://answers.opencv.org/question/1066/how-to-intuitively-interpret-gabor-lambda-param/
kernelSize = (5,5)
sigma = 5
theta = [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5]
lambd = 5
gamma = 1
psi = 0
gKernel = [cv2.getGaborKernel(kernelSize, sigma, deginrad(theta[i]), lambd, gamma, psi)for i in range(len(theta))]

kernel3 = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]) #Sharpening filter
thresholdLower = 50
thresholdHigher = 100

def preTreatment(picPath):
    """Load and apply transforms to the picture

    Keyword arguments:
        picPath -- path to the picture to load"""

    pic = cv2.imread(picPath)
    hsv = cv2.cvtColor(pic, cv2.COLOR_BGR2HSV)  # https://i.stack.imgur.com/TSKh8.png
    _, saturation, _ = cv2.split(hsv)
    _, mask = cv2.threshold(saturation, thresholdLower, thresholdHigher, cv2.THRESH_BINARY)
    pic = cv2.bitwise_and(pic, pic, mask=mask)
    pic = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
    pic = cv2.filter2D(pic, -1, kernel3)
    for j, kernel in enumerate(gKernel):
        test = cv2.filter2D(pic, -1, kernel)
        test = test // 8
        if j == 0:
            sumPic = test
        else:
            sumPic = cv2.add(sumPic, test)
    pic = sumPic
    if pic.shape[1] < pic.shape[0]:
        pic = cv2.resize(pic, resolution)
    else:
        pic = cv2.resize(pic, rotated_resolution)
    return pic



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


def createHistogram(sortedDescriptors, matcherRef):
    """Create an histogram of visual words for all of the pictures

    returns:
        allHistograms -- an array which contains histograms of every pictures
        allClasses -- an array which contains the classes for all histograms

    Keyword arguments:
        sortedDescriptors -- a dictionnary with key: class of the picture
                                                value: descriptors for every pictures of this class
        matcherRef -- a nearest neighbour matcher, which match given datas with the visual words"""

    allHistograms = []
    allClasses = []
    for key, value in sortedDescriptors.items():
        for imgDescriptors in value:
            hist = np.zeros(clustersNumber)
            meanDistRatio = 0
            for des in imgDescriptors:
                distances, indexes = matcherRef.kneighbors([des])
                distances = distances[0]
                ind = indexes[0][0]
                distRatio = distances[0]/distances[1]
                meanDistRatio+=distRatio
                if distRatio < acceptableDist:
                    hist[ind]+=1
            meanDistRatio/=len(imgDescriptors)
            #print(meanDistRatio)
            allHistograms.append(hist)
            allClasses.append(key)
    allHistograms = np.asarray(allHistograms)
    return allHistograms, allClasses

def nearestHist(testHist, refClasses, matcher):
    """Give, for a given histogram, the class of the nearest reference histogram

    returns:
        nearestClass -- class of the nearest reference histogram
        confidence -- confidence rate of this result

    Keyword arguments:
        testHist -- The histogram of the picture we want to classify
        refClasses -- an array which contains the classes for all reference histograms
        matcher -- A nearest neighbour matcher, which match given dats with reference histograms"""

    distances, indices = matcher.kneighbors([testHist])
    distances = distances[0]
    indices = indices[0]
    confidence = distances[0]/distances[1]
    nearestClass = refClasses[indices[0]]
    otherClass = refClasses[indices[1]]
    if  confidence > acceptableConfidence and nearestClass==otherClass:
        return (nearestClass, confidence)
    else:
        return (None, confidence)

def loadDB():
    """Compute BOVW, and create reference DB and corresponding matchers

    returns:
        matcherRef -- a nearest neighbour matcher, which match given datas with the visual words
        refClasses -- an array which contains the classes for all reference histograms
        matcher -- A nearest neighbour matcher, which match given dats with reference histograms

    Keyword arguments:
        None"""

    print("pic loading")
    sortedRefDes, allRefDes = loadAllPic(pathToTrainDir)
    print("clustering, this will take a while...")
    visualWords = kmeans(clustersNumber, allRefDes)
    print("matcher creation")
    matcherRef = NearestNeighbors(n_neighbors=2, algorithm='auto', metric='euclidean').fit(visualWords)
    print("creating histo")
    refHist, refClasses = createHistogram(sortedRefDes, matcherRef)
    print("matcher creation")
    matcher = NearestNeighbors(n_neighbors=2, algorithm='auto', metric='euclidean').fit(refHist)
    print("prep done.")
    return matcherRef, refClasses, matcher

if __name__ == '__main__':

    matcherRef, refClasses, matcher = loadDB()

    start = time()
    sortedValDes, _ = loadAllPic(pathToValDir)
    valHist, valClasses = createHistogram(sortedValDes, matcherRef)
    results = []
    for i, hist in enumerate(valHist):
        nearest, _ = nearestHist(hist, refClasses, matcher)
        results.append(nearest)
    end = time()

    precision = 0
    noneCounter = 0
    trapsAvoided = 0
    for i in range(len(results)):
        if results[i] is not None:
            if results[i]==valClasses[i]:
                precision+=1
        else:
            if valClasses[i]=="Traps":
                trapsAvoided+=1
            else:
                noneCounter+=1

    y_actu = pd.Series(valClasses, name='Actual')
    y_pred = pd.Series(results, name='Predicted')
    df_confusion = pd.crosstab(y_actu, y_pred)
    plt.matshow(df_confusion)
    plt.show()

    precision = precision / (len(results) - noneCounter - trapsAvoided)
    noneCounter = noneCounter / len(results)
    averageTime = (end - start) / len(valHist)
    trapsAvoided = trapsAvoided / len(os.listdir(pathToValDir + '/Traps'))
    print("Results: \n"
          "precision rate on acceptable picture: {:.4f}\n"
          "Non-acceptable picture rate: {:.4f}\n"
          "Traps detection rate: {:.4f}\n"
          "It took {} seconds per picture."
          .format(precision, noneCounter, trapsAvoided, averageTime))