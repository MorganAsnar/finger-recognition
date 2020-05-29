import numpy as np
import cv2
from collections import Counter


def deginrad(degree):
    """Take a degree angle, return the gradiant value"""
    radiant = 2*np.pi/360 * degree
    return radiant

matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)

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
resolution = (300,400)
rotated_resolution = (400,300)
acceptableConfidence = 0.75
acceptableDist = 1

def createHistogram(sortedDescriptors, matcherRef, clustersNumber):
    """Create an histogram of visual words for all of the pictures

    returns:
        allHistograms -- an array which contains histograms of every pictures
        allClasses -- an array which contains the classes for all histograms

    Keyword arguments:
        sortedDescriptors -- a dictionnary with key: class of the picture
                                                value: descriptors for every pictures of this class
        matcherRef -- a nearest neighbour matcher, which match given datas with the visual words
        clustersNumber -- number of centroids"""

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
    if  confidence > acceptableConfidence:
        return (nearestClass, confidence)
    else:
        return (None, confidence)


def preTreatment(picPath):
    """Load and apply transforms to the picture

    Keyword arguments:
        picPath -- path to the picture to load"""

    pic = cv2.imread(picPath)
    if pic is not None:
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

    else:
        return None