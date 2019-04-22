import os
import csv
import random
import math
import operator
from collections import namedtuple, defaultdict

"""
following along with
https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/
to implement K-nearest neighbors.

format of iris.data:
➜  knn head iris.data
5.1,3.5,1.4,0.2,Iris-setosa
4.9,3.0,1.4,0.2,Iris-setosa
4.7,3.2,1.3,0.2,Iris-setosa
4.6,3.1,1.5,0.2,Iris-setosa
"""

Dataset = namedtuple('Dataset', ['training_set', 'test_set'])
Neighbor = namedtuple('Neighbor', ['instance', 'distance'])

def loadDataset(filename, split):
    """
    loads dataset from filename.
    splits dataset according to given split ratio (decimal proportion)
    resulting in split proportion being allocated to the training set.

    returns `Dataset` containing `training_set` and `test_set
    """
    training_set = []
    test_set = []
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset) - 1):
            for y in range(4):
                dataset[x][y] = float(dataset[x][y])
            if random.random() < split:
                training_set.append(dataset[x])
            else:
                test_set.append(dataset[x])
    return Dataset(training_set=training_set, test_set=test_set)

def euclideanDistance(instance1, instance2, length):
    """
    calculates euclidean distance between two instances of arbitrary dimensionality.
    the instances are assumed to be lists of measurements.
    euclidean distance uses length number of dimensions to calculate.
    """
    def squared_difference(first, second):
        return (first - second) ** 2

    summed_squares = 0
    for x in range(length):
        summed_squares += squared_difference(first=instance1[x], second=instance2[x])
    return math.sqrt(summed_squares)

def getNeighbors(training_set, test_instance, k):
    """
    given a training set and test instance, returns the k-nearest neighbors from the
    test instance
    """
    # distances will become a list of tuples
    # each tuple containing (a_row_of_from_training_set, distance_from_test_instance)
    neighbors = []
    length = len(test_instance) - 1
    for training_instance in training_set:
        distance = euclideanDistance(test_instance, training_instance, length)
        neighbors.append(Neighbor(instance=training_instance, distance=distance))
    # fun fact: python list sort uses TimSort
    # https://en.wikipedia.org/wiki/Timsort
    # https://hg.python.org/cpython/file/tip/Objects/listsort.txt
    neighbors.sort(key=lambda neighbor: neighbor.distance)
    k_nearest = neighbors[0:k]
    return k_nearest

def getResponse(neighbors):
    """
    condense results via a vote
    assuming the class is the last attribute of each neighbor,
    tallies up the occurrences of each class
    and returns the class with the highest vote
    """
    class_votes = defaultdict(int)
    for neighbor in neighbors:
        instance_class = neighbor.instance[-1]
        class_votes[instance_class] += 1
    winner = max(class_votes.keys(), key=(lambda class_vote_key: class_votes[class_vote_key]))
    return winner

def getAccuracy(test_set, predictions):
    correct = [test_instance for index, test_instance in enumerate(test_set) if test_instance[-1] == predictions[index]]
    accuracy = 100 * float(len(correct)) / float(len(test_set))
    return accuracy

def main():
    # prepare data
    dataset = loadDataset(
        filename=os.path.join(os.path.dirname(__file__), 'iris.data'),
        split=0.67
        )
    training_set, test_set = dataset.training_set, dataset.test_set
    print('Training set len: ' + repr(len(training_set)))
    print('Test set len: ' + repr(len(test_set)))
    # generate predictions
    predictions=[]
    k = 3
    print(f"k: {k}")
    for test_instance in test_set:
        neighbors = getNeighbors(
            training_set=training_set,
            test_instance=test_instance,
            k=k
            )
        result = getResponse(neighbors=neighbors)
        predictions.append(result)
        print(f"""
    prediction: {result}
    actual: {test_instance[-1]}
    correct: {result == test_instance[-1]}
            """)
    accuracy = getAccuracy(test_set=test_set, predictions=predictions)
    print(f"accuracy: {accuracy}%")

if __name__ == '__main__':
    main()
