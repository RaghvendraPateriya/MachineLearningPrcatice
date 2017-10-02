from numpy import *
import operator


# DataSet & Label Initialization Section
def createDataSet():
    group = array ([
            [1.0, 1.1],
            [1.0, 1.0],
            [0, 0],
            [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


# Classifier
def classify0(inX, dataSet, label, k):
    '''
    Here we are going to calculating diatsance using Euclidian Distance.
    '''
    dataSetSize = dataSet.shape[0]
    print(dataSetSize)
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistance = sqDiffMat.sum(axis=1)
    distances = sqDistance**0.5
    sortedDistIndices = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = label[sortedDistIndices[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1),
                              reverse=True)
    return sortedClassCount[0][0]

g, l = createDataSet()

print (g, l)

input = [0, 0]

print(classify0(input, g, l, 3))

