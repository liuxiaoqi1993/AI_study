#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import random
#构造格式化数据 形如w0*x0+w1*x1······wn*xn = Z 第一项为常数项
def loadDataSet():
    dataSet = []; labelSet = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataSet.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelSet.append(float(lineArr[2]))
    return dataSet,labelSet
#构造阶跃函数，因为后面的梯度下降算法会用到矩阵，这里用numpy的exp函数。ps 别用math
def sigmoid(inX):
    return 1.0/(1+np.exp(-inX))

#使用梯度下降/上升.
def gradAscent(dataSet,labelSet):
    #数据转换为矩阵
    inputMatrix = np.mat(dataSet)
    classMatrix = np.mat(labelSet).transpose()#转置，方便后续进行计算
    #for i in range(次数): w = w +/- α*▽wf(w)
    maxCircles = 500#设置最大迭代次数
    alpha = 0.001#步长
    m,n = inputMatrix.shape
    weights = np.ones((n,1))#初始化 W
    for i in range(maxCircles):
        #(sigmoid(X*W) - y)*X   具体步骤请用草稿写一下，一目了然
        h = sigmoid(inputMatrix*weights)#这里为矩阵 *
        error = classMatrix - h  #[e0,e1,.....,en]T  e中的每个元素*对应样本中的第i个元素，然后累加，即为梯度的第i个分量
        weights = weights + alpha*inputMatrix.transpose()*error   #n*m * m*1   =  n*1
    return weights


def BestFit(wei):
    weight = np.array(wei) ##矩阵转数组
    dataSet, labelSet = loadDataSet()
    dataArr = np.array(dataSet)
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(len(labelSet)):
        if int(labelSet[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weight[0] - weight[1]*x)/weight[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

def stocGardAscent(dataSet, labelSet, numIter=150):
    m, n = np.shape(dataSet)
    we = np.ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0 + j + i) + 0.01
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataSet[randIndex] * we))
            error = labelSet[randIndex] - h
            we = we + alpha * dataSet[randIndex] * error
            #del(dataIndex[randIndex])
    return we

def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5:
        return 1
    else:
        return 0

def colicTest():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []; trainingLabel = []
    for line in frTrain.readlines():
        lineArr = line.strip().split('\t')
        trainingSet.append([float(i) for i in lineArr[:-2]])
        trainingLabel.append(float(lineArr[-1]))
    testSet = []
    testLabel = []
    for line in frTest.readlines():
        lineArr = line.strip().split('\t')
        testSet.append([float(i) for i in lineArr[:-2]])
        testLabel.append(float(lineArr[-1]))
    weights = stocGardAscent(np.array(trainingSet), trainingLabel, numIter=1000)
    error = 0.0
    for i in range(len(testSet)):
        if classifyVector(np.array(testSet[i]),weights) != testLabel[i]:
            error += 1
    return error/float(len(testSet))

def multiTese():
    num = 10; sum = 0.0
    for i in range(num):
        sum += colicTest()
    return sum/num
print(multiTese())
# dataSet, labelSet = loadDataSet()
# weig = stocGardAscent(np.array(dataSet), labelSet)
# BestFit(weig)





