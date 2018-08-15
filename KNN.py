#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import numpy as np
import operator
import matplotlib
import matplotlib.pyplot as plt
from os import listdir
#demo2
def classify0(inX, dataSet, labels, k):
    datalength = dataSet.shape[0]
    diffMat = np.tile(inX,(datalength,1)) - dataSet
    sqdist = diffMat**2
    sqdistances = sqdist.sum(axis=1)
    distances = np.sqrt(sqdistances)
    sortdeDis = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortdeDis[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassLabels = max(classCount.items(),key=operator.itemgetter(1))
    #print(classCount)

    return sortedClassLabels[0]



def file2matrix(filename):
    love_dictionary={'largeDoses':3, 'smallDoses':2, 'didntLike':1}
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)            #get the number of lines in the file
    returnMat = np.zeros((numberOfLines,3))        #prepare matrix to return
    classLabelVector = []                       #prepare labels return
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        if(listFromLine[-1].isdigit()):
            classLabelVector.append(int(listFromLine[-1]))
        else:
            classLabelVector.append(love_dictionary.get(listFromLine[-1]))
        index += 1
    fr.close()
    return returnMat,classLabelVector


# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(dataMat[:,1],dataMat[:,2],15*np.array(classLabelVector),15*np.array(classLabelVector))
# plt.show()

#归一化 newValue = (oldValue - minValue)/(maxValue - minValue)
def autoNorm(dataSet):
    minValues = dataSet.min(0)
    maxValues = dataSet.max(0)
    ranges = maxValues - minValues
    m = dataSet.shape[0]
    normValue = dataSet - np.tile(minValues,(m,1))
    normValue = normValue/np.tile(ranges,(m,1))
    return  normValue,ranges,minValues

def datingClassTest():
    hoRatio = 0.10
    fileMat,classLabelVector = file2matrix('datingTestSet2.txt')
    dataNormMat,ranges,minValues = autoNorm(fileMat)
    m = dataNormMat.shape[0]
    numTestVecs = m * hoRatio
    testDate = dataNormMat[:int(numTestVecs)]
    dataSet = dataNormMat[int(numTestVecs):]
    index = 0
    errorCount = 0.0
    for inX in testDate:
        testLabel = classify0(inX, dataSet, classLabelVector[int(numTestVecs):], 3)
        print(testLabel)
        if  testLabel != classLabelVector[index]:
            errorCount += 1
        index += 1
    print(errorCount/float(numTestVecs))

    # errorcount = 0.0
    # for i in range(int(numTestVecs)):
    #     testLabel = classify0(dataNormMat[i],dataNormMat[int(numTestVecs):],classLabelVector[int(numTestVecs):],3)
    #     if testLabel != classLabelVector[i]:
    #         errorcount += 1
    # errorRatio = errorcount/float(numTestVecs)
    # print(errorcount)
    # print(errorRatio)

# def datingClassTest():
#     hoRatio = 0.10      #hold out 10%
#     datingDataMat,datingLabels = file2matrix('datingTestSet.txt')       #load data setfrom file
#     normMat, ranges, minVals = autoNorm(datingDataMat)
#     m = normMat.shape[0]
#     numTestVecs = int(m*hoRatio)
#     errorCount = 0.0
#     for i in range(numTestVecs):
#         classifierResult = classify0(normMat[i],normMat[numTestVecs:],datingLabels[numTestVecs:],3)
#
#         if (classifierResult != datingLabels[i]): errorCount += 1.0
#     print("the total error rate is: %f" % (errorCount/float(numTestVecs)))
#     print(errorCount)


def img2vector(filename):
    k = 32
    fr = open(filename)
    arrayOLines = fr.readlines()
    resultVector = np.zeros(1024)
    for i in range(k):
        for j in range(k):
            resultVector[i * k + j] = int(arrayOLines[i][j])
    return resultVector

def handwritingclassTest():
    traindirs = listdir('digits/trainingDigits')
    m = len(traindirs)
    hwlabels = []
    #导入训练数据
    trainMat = np.zeros((m,1024))
    for i in range(m):
        trainMat[i] = img2vector('digits/trainingDigits/'+traindirs[i])
        splitdir = traindirs[i].split('.')[0]
        splitclass = splitdir.split('_')[0]
        hwlabels.append(int(splitclass[0]))

    testdirs = listdir('digits/testDigits')
    n = len(testdirs)
    testhwlabels = []
    testMat = np.zeros((n,1024))
    for i in range(n):
        testMat[i] = img2vector('digits/testDigits/'+testdirs[i])
        splitdir = testdirs[i].split('.')[0]
        splitclass = splitdir.split('_')[0]
        testhwlabels.append(int(splitclass[0]))
    errorCount = 1
    for i in range(n):
        if testhwlabels[i] != classify0(testMat[i], trainMat, hwlabels, 3):
            errorCount += 1
    print(errorCount/float(n))

handwritingclassTest()

