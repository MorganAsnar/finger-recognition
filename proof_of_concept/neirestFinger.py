import numpy as np
import cv2
from collections import Counter


matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
#matcher = cv2.BFMatcher()


def nearestFeatures(query, ref, featuresIndex):
    indexes = []
    nearest = []
    matches = matcher.knnMatch(query, ref, k=2)
    for m, n in matches:
        if m.distance < 0.75*n.distance:
            refIndex = m.trainIdx
            #dist = m.distance
            indexes.append(featuresIndex[refIndex])
    identifiedClasses = Counter(indexes)
    if len(indexes) > 0:
        confidenceLevel = identifiedClasses.most_common(1)[0][1]/len(indexes)
    try:
        return (identifiedClasses.most_common(1)[0][0], confidenceLevel)
    except:
        return None