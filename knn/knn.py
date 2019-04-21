import csv
import random
import math
import operator

"""
following along with
https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/
to implement K-nearest neighbors.

todo: fix up the example code

format of iris.data:
➜  knn head iris.data
5.1,3.5,1.4,0.2,Iris-setosa
4.9,3.0,1.4,0.2,Iris-setosa
4.7,3.2,1.3,0.2,Iris-setosa
4.6,3.1,1.5,0.2,Iris-setosa
"""

def loadDataset(filename, split, trainingSet=[] , testSet=[]):
    """
    loads dataset from filename.
    splits dataset according to given split ratio (decimal proportion)
    resulting in split proportion being allocated to the training set.

    training and test set args are mutated to hold the allocated data.
    todo: this function should probably return the split data set.
    """
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset) - 1):
            for y in range(4):
                dataset[x][y] = float(dataset[x][y])
            if random.random() < split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])

def euclideanDistance(instance1, instance2, length):
    """
    calculates euclidean distance between two instances of arbitrary dimensionality.
    the instances are assumed to be lists of measurements.
    euclidean distance uses length number of dimensions to calculate.
    """
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)

def getNeighbors(trainingSet, testInstance, k):
    """
    given a training set and test instance, returns the k-nearest neighbors from the
    test instance
    """
    # distances will become a list of tuples
    # each tuple containing (a_row_of_from_training_set, distance_from_test_instance)
    distances = []
    length = len(testInstance) - 1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    # fun fact: python list sort uses TimSort
    # https://en.wikipedia.org/wiki/Timsort
    # https://hg.python.org/cpython/file/tip/Objects/listsort.txt
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

def getResponse(neighbors):
    """
    condense results via a vote
    assuming the class is the last attribute of each neighbor,
    tallies up the occurrences of each class
    and returns the class with the highest vote
    """
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]

def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct/float(len(testSet))) * 100.0

def main():
    # prepare data
    trainingSet=[]
    testSet=[]
    split = 0.67
    loadDataset('iris.data', split, trainingSet, testSet)
    print('Train set len: ' + repr(len(trainingSet)))
    print('Test set len: ' + repr(len(testSet)))
    # generate predictions
    predictions=[]
    k = 3
    print(f"k: {k}")
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainingSet, testSet[x], k)
        result = getResponse(neighbors)
        predictions.append(result)
        print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
    accuracy = getAccuracy(testSet, predictions)
    print('Accuracy: ' + repr(accuracy) + '%')

if __name__ == '__main__':
    main()
