# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 23:06:36 2015

@author: Brian
"""

import cv2
import numpy as np
import os

image_paths = []
path = "C:\Python27\Project2\TestingSet"

#list of our class names
training_names = os.listdir(path)

training_paths = []
names_path = []
#get full list of all training images
for p in training_names:
    training_paths1 = os.listdir("C:\Python27\Project2\TestingSet\\"+p)
    for j in training_paths1:
        training_paths.append("C:\Python27\Project2\TestingSet\\"+p+"\\"+j)
        names_path.append(p)

sift = cv2.SIFT()
print names_path

descriptors_unclustered = []

dictionarySize = 5

BOW = cv2.BOWKMeansTrainer(dictionarySize)

for p in training_paths:
    image = cv2.imread(p)
    gray = cv2.cvtColor(image, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    kp, dsc= sift.detectAndCompute(gray, None)
    BOW.add(dsc)

#dictionary created
dictionary = BOW.cluster()


FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params,search_params)
sift2 = cv2.DescriptorExtractor_create("SIFT")
bowDiction = cv2.BOWImgDescriptorExtractor(sift2, cv2.BFMatcher(cv2.NORM_L2))
bowDiction.setVocabulary(dictionary)
print "bow dictionary", np.shape(dictionary)


#returns descriptor of image at pth
def feature_extract(pth):
    im = cv2.imread(pth, 1)
    gray = cv2.cvtColor(im, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    return bowDiction.compute(gray, sift.detect(gray))

train_desc = []
train_labels = []
i = 0
for p in training_paths:
    train_desc.extend(feature_extract(p))
    if names_path[i]=='Chair':
        train_labels.append(1)
    if names_path[i]=='Doge':
        train_labels.append(2)
    if names_path[i]=='Football':
        train_labels.append(3)
    if names_path[i]=='Watch':
        train_labels.append(4)
    if names_path[i]=='Wrench': 
        train_labels.append(5)
    i = i+1

print "svm items", len(train_desc), len(train_desc[0])
count=0
svm = cv2.SVM()
svm.train(np.array(train_desc), np.array(train_labels))
i=0
j=0

confusion = np.zeros((5,5))
def classify(pth):
    feature = feature_extract(pth)
    p = svm.predict(feature)
    confusion[train_labels[count]-1,p-1] = confusion[train_labels[count]-1,p-1] +1
    
    

for p in training_paths:
    classify(p)
    count+=1

def normalizeRows(M):
    row_sums = M.sum(axis=1)
    return M / row_sums
    
confusion = normalizeRows(confusion)

confusion = confusion.transpose()
    
print confusion