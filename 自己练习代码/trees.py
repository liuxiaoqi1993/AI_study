#!/usr/bin/env python 
# -*- coding:utf-8 -*-
from math import log

def calshannonEnt(dataSet):
    numEntries = len(dataSet)
    labelsCount = {}
    for i in range(numEntries):
        currentVec = dataSet[i]
        labelsCount[currentVec[-1]] = labelsCount.get(currentVec[-1],0) + 1
    shannonEnt = 0.0
    for key in labelsCount:
        prob = float(labelsCount[key])/numEntries
        shannonEnt -= prob*log(prob,2)
    return shannonEnt

def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if value == featVec[axis]:
            feat = featVec[:axis]
            feat.extend(featVec[axis+1:])
            retDataSet.append(feat)
    return retDataSet


def choosenBestFeature2split(dataSet):

    #计算数据集香农Ent
    EntD = calshannonEnt(dataSet)
    numD = len(dataSet)
    #划分按照属性数据集
    numFeat = len(dataSet[0]) - 1
    entSet = {}
    for i in range(numFeat):
        #列表生成式
        valueFeat = [exp[i] for exp in dataSet]
        uniFeatValue = set(valueFeat)
        # for featVec in dataSet:
        #     valueFeat[featVec[i]] = featVec[i]
        wEnt = 0.0
        for value in uniFeatValue:
            # 计算子集的权重ent
            childDataSet = splitDataSet(dataSet,i,value)
            numV = len(childDataSet)
            wEnt += (numV/numD)*calshannonEnt(childDataSet)
        # 计算单个属性的信息增益
        gain = EntD - wEnt
        entSet[i] = gain
    feature = max(entSet.items(),key=lambda s:s[1])
    #返回特征属性

    return feature[0]


def majorityCnt(classList):
    labelsCount = {}
    for label in classList:
        labelsCount[label] = labelsCount.get(label,0) + 1
    retLabel = max(labelsCount.items(),lambda s:s[1])
    return retLabel[0]


def createTrees(dataSet,labels):
    #返回条件
    classLabel = []
    for label in dataSet:
        classLabel.append(label[-1])
    if len(set(classLabel)) == 1:
        return classLabel[0]
    if (len(dataSet[0]) == 1):

        return majorityCnt(classLabel)
    bestfeature = choosenBestFeature2split(dataSet)
    valueFeat = [exp[bestfeature] for exp in dataSet]
    uniFeatValue = set(valueFeat)
    mytree = {}
    tmp = labels[:]
    tmp.pop(bestfeature)
    childtree = {}
    for value in uniFeatValue:
        childSet = (splitDataSet(dataSet, bestfeature, value))
        if len(childSet) == 0:
            return {value:majorityCnt(classLabel)}
        node = createTrees(childSet,tmp)
        childtree[value] = node
    mytree[labels[bestfeature]] = childtree

    return mytree


def createDataSet():
    dataSet = [[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']]
    labels = ['no surfacing','flipers']
    return dataSet,labels
da,la = createDataSet()
print(createTrees(da,la))