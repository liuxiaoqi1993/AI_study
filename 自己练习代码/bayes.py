#!/usr/bin/env python 
# -*- coding:utf-8 -*-
from numpy import *
import random
import feedparser
def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', \
                    'problems', 'help', 'please'],\
                   ['maybe', 'not', 'take', 'him',\
                    'to','dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute',\
                    'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'steak', 'how', \
                    'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]
    return postingList,classVec

def createVocabList(dataSet):
    vocabList = set([])
    for data in dataSet:
        vocabList = set(data)|vocabList
    return list(vocabList)

def setOfWord2Vec(vocabList,inputSet):

    returnVec = [0] * len(vocabList)
    for word in vocabList:
        if word in inputSet:
            returnVec[vocabList.index(word)] = 1
        else:
            returnVec[vocabList.index(word)] = 0
    return returnVec

# print(vocabList)
# print(setOfWord2Vec(vocabList,dataSet[0]))



def trainNB0(trainMatrix, classCatogery):
    cateNum = len(classCatogery)
    num1 = sum(classCatogery)
    P1c = num1/cateNum
    trainWordNum = len(trainMatrix[0])
    p0 = ones(trainWordNum); p1 = ones(trainWordNum)
    p1demon = 2.0; p0demon = 2.0
    for i in range(cateNum):
        if classCatogery[i] == 1: #1 代表侮辱 0代表正常
            p1 += trainMatrix[i]
            p1demon += sum(trainMatrix[i])
        else:
            p0 += trainMatrix[i]
            p0demon += sum(trainMatrix[i])
    p1Vec = log(p1/p1demon)
    p0Vec = log(p0/p0demon)
    return p0Vec, p1Vec, P1c



def classifyNB(vec2classify, p0Vec, p1Vec, P1c):
    p0 = sum(vec2classify*p0Vec)+log(1-P1c)
    p1 = sum(vec2classify*p1Vec)+log(P1c)
    if p0 > p1:
        return 0
    else:
        return 1

def testingNb():
    dataSet, classLabels = loadDataSet()
    vocabList = createVocabList(dataSet)
    trainMatrix = []
    for word in dataSet:
        trainMatrix.append(setOfWord2Vec(vocabList, word))
    p0, p1, p1c = trainNB0(trainMatrix, classLabels)
    testEntry = ['stupid', 'garbage']
    thisDoc = setOfWord2Vec(vocabList, testEntry)
    return classifyNB(thisDoc, p0, p1, p1c)

def bagOfWord2Vec(vocabList,inputSet):

    returnVec = [0] * len(vocabList)
    for word in vocabList:
        if word in inputSet:
            returnVec[vocabList.index(word)] = 1
        else:
            returnVec[vocabList.index(word)] = 0
    return returnVec

def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W', bigString)
    return [token.lower() for token in listOfTokens if len(token) > 2]

def spamTest():
    import random
    docList = [];classList = [];trainMatrix = []
    for i in range(1,26):
        a = open('email/ham/%d.txt' % i).read()

        wordlist = textParse(a)

        docList.append(wordlist)

        classList.append(0)
        docList.append(textParse(open('email/spam/%d.txt'%i).read()))
        classList.append(1)
    vocabSet = createVocabList(docList)

    for i in range(50):
        trainMatrix.append(bagOfWord2Vec(vocabSet,docList[i]))

    trainSetindex = list(range(50)); TestSetindex = []
    for i in range(10):
        index = int(random.uniform(0, len(trainSetindex)))
        TestSetindex.append(trainSetindex[index])
        del(trainSetindex[index])
    trainMat = [trainMatrix[i] for i in trainSetindex]
    classVec = [classList[i] for i in trainSetindex]
    p0, p1, p1c = trainNB0(array(trainMat), array(classVec))
    error = 0.0
    for index in TestSetindex:
        if classifyNB(array(trainMatrix[index]), p0, p1, p1c) != classList[index]:
            error += 1
    return error/float(len(TestSetindex))

def calMostFreq(fullText):
    freDict = {}
    for i in range(len(fullText)):
        freDict[fullText[i]] = freDict.get(fullText[i], 0) + 1
    sortedFreq = sorted(freDict.items(), key=lambda s:s[1],reverse=True)
    return sortedFreq[:10]

def localWords(feed1, feed0):

    docList = []; classList = []; fullText = []
    minlen = min(len(feed1['entries']), len(feed0['entries']))

    for i in range(minlen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    top5Words = calMostFreq(fullText)
    for pairW in top5Words:
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0])
    trainSet = list(range(2*minlen)); testSet = []
    for i in range(5):
        index = int(random.uniform(0,len(trainSet)))
        testSet.append(index)
        del(trainSet[index])
    trainMatrix = []
    classSet = []
    for i in trainSet:
        trainMatrix.append(bagOfWord2Vec(vocabList, docList[i]))
        classSet.append(classList[i])

    p0,p1,p1c = trainNB0(array(trainMatrix), array(classSet))
    errorCount = 0.0
    for j in testSet:
        wordVect = bagOfWord2Vec(vocabList, docList[j])
        if classifyNB(array(wordVect), p0, p1, p1c) != classList[j]:
            errorCount += 1
    print(errorCount/len(testSet))
    return vocabList, p0, p1


ny = feedparser.parse('http://www.nasa.gov/rss/dyn/image_of_the_day.rss')
sf = feedparser.parse('http://rss.yule.sohu.com/rss/yuletoutiao.xml')


def getTopWords(ny, sf):
    vocabList, p0, p1 = localWords(ny, sf)

    top0 = {}; top1 = {}
    for i in range(len(p0)):
        if p0[i] > -6: top0[vocabList[i]] = p0[i]
        if p1[i] > -6: top1[vocabList[i]] = p1[i]
    st0 = sorted(top0.items(), key=lambda s: s[1], reverse=True)
    st1 = sorted(top1.items(), key=lambda s: s[1], reverse=True)
    for item in st0:
        print(item[0])
    print('-------------------------------------------------------------')
    for item in st1:
        print(item[0])
getTopWords(ny, sf)