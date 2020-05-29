import pickle
import cv2
import sys
from fingerUtil import preTreatment, createHistogram, nearestHist

clustersNumber = 750

if __name__ == '__main__':

    try:
        matcher = pickle.load(open('matcherPickle_file', 'rb'))
        matcherRef = pickle.load(open('matcherRefPickle_file', 'rb'))
        refClasses = pickle.load(open('refClassesPickle_file.pickle', 'rb'))
    except:
        sys.exit('Database not found. Considere running CreateDB.py first, '
                 'or check the path to the save directory.')

    sift = cv2.xfeatures2d.SIFT_create()
    nearest = None
    path = sys.argv[1].replace("\\", "\\\\")
    picture = preTreatment(path)
    if picture is not None:
        _, des = sift.detectAndCompute(picture, None)
        queryDes = {}
        queryDes["queryPic"] = [des]
        queryHist, _ = createHistogram(queryDes, matcherRef, clustersNumber)
        queryHist = queryHist[0]
        nearest, confidence = nearestHist(queryHist, refClasses, matcher)
        if nearest is not None:
            print("This finger belongs to {}.\n"
                  "Confidence rate : {}%".format(nearest, confidence*100))
        else:
            print("This finger can't be identified properly.")
    else :
        sys.exit('Invalid path')

