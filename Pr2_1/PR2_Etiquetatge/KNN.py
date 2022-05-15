__authors__ = ['1647904']
__group__ = 'DM.10'

import numpy as np
import math
import operator
from scipy.spatial.distance import cdist

class KNN:
    def __init__(self, train_data, labels):

        self._init_train(train_data)
        self.labels = np.array(labels)
        self.neighbours = []
        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################


    def _init_train(self,train_data):
        """
        initializes the train data
        :param train_data: PxMxNx3 matrix corresponding to P color images
        :return: assigns the train set to the matrix self.train_data shaped as PxD (P points in a D dimensional space)
        """
        # ensures that the values are of the float type;
        if train_data.dtype != float:
            train_data = train_data.astype(float)

        shape = train_data.shape
        self.train_data = np.reshape(train_data, (shape[0], shape[1] * shape[2] * shape[3]))

    def get_k_neighbours(self, test_data, k):
        """
        given a test_data matrix calculates de k nearest neighbours at each point (row) of test_data on self.neighbors
        :param test_data:   array that has to be shaped to a NxD matrix ( N points in a D dimensional space)
        :param k:  the number of neighbors to look at
        :return: the matrix self.neighbors is created (NxK)
                 the ij-th entry is the j-th nearest train point to the i-th test point
        """

        shape = test_data.shape
        points = np.reshape(test_data, (shape[0], shape[1] * shape[2] * shape[3]))
        distances = cdist(points, self.train_data)

        for point_distances in distances:
            train_indexes = list(range(len(self.train_data)))
            self.get_k_neighbours_quicksort(train_indexes, point_distances, 0, len(point_distances) - 1)
            nearest_neighbours = train_indexes
            k_nearest_neighbours = nearest_neighbours[:k]
            self.neighbours.append([self.labels[index] for index in k_nearest_neighbours])

    def get_k_neighbours_quicksort(self, train_points, distances, low, high):

        if len(train_points) == 1:
            return train_points

        if low < high:
            # pi is partitioning index, arr[p] is now
            # at right place
            pi = self.get_k_neighbours_partition(train_points, distances, low, high)

            # Separately sort elements before
            # partition and after partition
            self.get_k_neighbours_quicksort(train_points, distances, low, pi - 1)
            self.get_k_neighbours_quicksort(train_points, distances, pi + 1, high)

    def get_k_neighbours_partition(self, points, distances, low, high):
        i = (low - 1)  # index of smaller element
        pivot = distances[high]  # pivot

        for j in range(low, high):
            # If current element is smaller than or
            # equal to pivot
            if distances[j] <= pivot:
                # increment index of smaller element
                i = i + 1
                points[i], points[j] = points[j], points[i]
                distances[i], distances[j] = distances[j], distances[i]

        points[i + 1], points[high] = points[high], points[i + 1]
        distances[i + 1], distances[high] = distances[high], distances[i + 1]
        return i + 1



    def get_class(self):
        """
        Get the class by maximum voting
        :return: 2 numpy array of Nx1 elements.
                1st array For each of the rows in self.neighbors gets the most voted value
                            (i.e. the class at which that row belongs)
                2nd array For each of the rows in self.neighbors gets the % of votes for the winning class
        """

        most_voted_labels = []
        possible_labels = sorted(list(set(self.labels)))
        for neighbour_points in self.neighbours:

            class_votes = [0 for _ in possible_labels]
            total_votes = 0

            for neighbour in neighbour_points:
                class_votes[list(possible_labels).index(neighbour)] += 1
                total_votes += 1

            # get most voted class
            max_votes = max(class_votes)
            class_index = class_votes.index(max_votes)
            most_voted_label = possible_labels[class_index]
            most_voted_labels.append(most_voted_label)

            # get votes percentage
            percentage_to_win = max_votes / total_votes
            print("Elected: ", most_voted_label, "with", str(round(percentage_to_win * 100, 2))+"% of the votes.")

        return most_voted_labels

    def predict(self, test_data, k):
        """
        predicts the class at which each element in test_data belongs to
        :param test_data: array that has to be shaped to a NxD matrix ( N points in a D dimensional space)
        :param k:         :param k:  the number of neighbors to look at
        :return: the output form get_class (2 Nx1 vector, 1st the classm 2nd the  % of votes it got
        """
        self.get_k_neighbours(test_data, k)
        return self.get_class()
