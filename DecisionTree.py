import numpy as np
import pandas as pd
import copy
from queue import Queue

condition=["None","color","root","knocks","texture","umbilicus","touch"]

vis=[]
feathearNum=0

def loadData(path):
    csv=pd.read_csv(path)
    csv=np.array(csv)
    return (csv[:,1:np.shape(csv)[1]],csv[:,-1])

def getFeathearNum(trX):
    FN=[]
    for i in range(np.shape(trX)[1]-1):
        atrr=[]
        for j in range(np.shape(trX)[0]):
            if trX[j][i] not in atrr:
                atrr.append(trX[j][i])
        FN.append(atrr)
    return FN

def chooseBestFeathear(trX,trY):
    global vis
    Ent=calcuCrossEntrophy(trY)
    sortList={}
    for i,item in enumerate(getFeathearNum(trX)):
        if i not in vis:
           ent=0
           for atrr in item:
              ent+=calcuCrossEntrophy(trY[trX[:,i]==atrr])
           sortList[i]=Ent-ent
    sortedList=sorted(sortList.items(), key=lambda d:d[1], reverse = True)
    return sortedList[0][0]

def calcuCrossEntrophy(classList):
    classCount={}
    sum=len(classList)
    crossEntrophy=0
    for vote in classList:
        if vote in classCount:
            classCount[vote]+=1
        else:
            classCount[vote]=1
    for key,value in classCount.items():
        crossEntrophy+=-(value/sum)*np.log(value/sum)
    return crossEntrophy
class Node:
    def __init__(self):
        self.cond=""
        self.children=[]
        self.isLeaves=0
        self.label=None

def buildDecisionTree(root,X,Y):
    global vis
    if(len(vis)==feathearNum):
        root.isLeaves=1
        good=np.sum(Y)
        root.label="Good" if (good)*1.0/len(Y)>=0.5 else "Bad"
        return
    if np.sum(Y)==len(Y)|np.sum(Y)==0:
        root.isLeaves = 1
        root.label = "Good" if np.sum(Y)!=0 else "Bad"
        return


    bestFeathear=chooseBestFeathear(X,Y)
    feathear=getFeathearNum(X)[bestFeathear]
    for index in feathear:
        temp=Node()
        temp.cond=index
        # print(condition[bestFeathear+1])
        root.label=bestFeathear
        root.children.append(temp)
        vis.append(bestFeathear)
        buildDecisionTree(temp,X[X[:,bestFeathear]==index,:],Y[X[:,bestFeathear]==index])
        vis.pop()


class decisionTree:
    def __init__(self):
        self.root=Node()
    def Train(self,trX,trY):
        buildDecisionTree(self.root,trX,trY)
    def predict(self,X):
        root=self.root
        while root.isLeaves==0:
            for child in root.children:
                if X[root.label]==child.cond:
                    root=child
                    break

        return root.label

    def display(self):
        que=Queue()
        que.put(self.root)
        while ~que.empty():
            root=que.get()
            if root.isLeaves==0:
                print(root.cond, condition[root.label])
                for child in root.children:
                    que.put(child)


pathTrain=r"C:\Users\34780\Desktop\大二下\机器学习\作业\实验三\朴素贝叶斯\watermelon3_0_En.csv"
pathTest=r"C:\Users\34780\Desktop\大二下\机器学习\作业\实验三\朴素贝叶斯\test.csv"
(trX,trY)=loadData(pathTrain)
(teX,teY)=loadData(pathTest)
feathearNum=np.shape(trX)[1]-1
dT=decisionTree()
dT.Train(trX,trY)
print(dT.predict([2,1,2,1,1,1]))
