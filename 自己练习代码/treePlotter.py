#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import trees
decisionNode = dict(boxstyle='sawtooth', fc='0.8')
leafNode = dict(boxstyle='round4', fc='0.8')
arrow_args = dict(arrowstyle='<-')

def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',\
                                xytext=centerPt, textcoords='axes fraction',\
                                va='center', ha='center', bbox=nodeType, arrowprops=arrow_args)

# def createPlot():
#     fig = plt.figure(1, facecolor='white')
#     fig.clf()
#     createPlot.ax1 = plt.subplot(111, frameon=False)
#     plotNode('a decison node',(0.5, 0.1), (0.1, 0.5), decisionNode)
#     plotNode('a leaf node', (0.8, 0.1), (0.3, 0.8), leafNode)
#     plt.show()


def getNumLeafs(myTree):
    numLeafs = 0
    secondDict = list(myTree.values())[0]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs


def getTreeDeepth(myTree):
    maxdepth = 0
    secondDict = list(myTree.values())[0]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDeepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxdepth: maxdepth=thisDepth
    return maxdepth


def retriveTree(i):
    listOfTrees = [{'no surfacing': {0: 'no',1: {'flippers':\
                                                     {0:'no',1:'yes'}},3:'Maybe'}}, \
                   {'no surfacing': {0: 'no', 1: {'flippers': \
                                                     {0:'no',1:'yes'}}}}]
    return listOfTrees[i]


def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString)

def plotTree(myTree, parentPt, nodeTxt):
    numLeafs = getNumLeafs(myTree)
    firstStr = list(myTree.keys())[0]
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW ,\
              plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key],cntrPt,str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key],(plotTree.xOff,plotTree.yOff),cntrPt,leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD

def plotTree2(myTree, parentPt, nodeTxt):
    #属性节点
    numLeafs = getNumLeafs(myTree)
    firstStr = list(myTree.keys())[0]
    cntrPt = (plotTree2.xOff + (1.0+numLeafs)/2.0/plotTree2.totalW, plotTree2.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    #递归画子树
    secondDict = myTree[firstStr]
    plotTree2.yOff -= 1/plotTree2.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            plotTree2(secondDict[key],cntrPt,str(key))
        else:
            plotTree2.xOff = plotTree2.xOff + 1.0 / plotTree2.totalW
            plotNode(secondDict[key],(plotTree2.xOff, plotTree2.yOff),cntrPt,leafNode)
            plotMidText((plotTree2.xOff, plotTree2.yOff),cntrPt,str(key))
    plotTree2.yOff += 1/plotTree2.totalD



def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111,frameon=False, **axprops)
    plotTree2.totalW = float(getNumLeafs(inTree))
    plotTree2.totalD = float(getTreeDeepth(inTree))
    plotTree2.xOff = -0.5/plotTree2.totalW; plotTree2.yOff = 1.0;
    plotTree2(inTree, (0.5,1.0), '')
    plt.show()



def classify(inputTree,featLabels,testVec):
    firstStr = list(inputTree.keys())[0]
    index = featLabels.index(firstStr)
    secondDict = inputTree[firstStr]
    classLael =  testVec[index]
    for key in secondDict.keys():
        if classLael == key:
            if type(secondDict[key]).__name__ == 'dict':
                return classify(secondDict[key], featLabels,testVec)
            else:
                return secondDict[key]

def createDataSet():
    dataSet = [[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']]
    labels = ['no surfacing','flippers']
    return dataSet,labels


def storeTree(inputTree, filename):
    import pickle
    fw = open(filename,'wb')
    pickle.dump(inputTree,fw)
    fw.close()

def loadTree(filename):
    import pickle
    fw = open(filename,'rb')
    tree = pickle.load(fw)
    fw.close()
    return tree

def testTree():
    fr = open('lenses.txt')
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    print(lenses)
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    myTree = trees.createTrees(lenses,lensesLabels)
    createPlot(myTree)
    print(myTree)

testTree()
#{'tearRate': {'reduced': 'no lenses', 'normal': {'astigmatic': {'no': {'age':