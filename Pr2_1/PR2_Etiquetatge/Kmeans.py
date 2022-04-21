__authors__ = ['1647904']
__group__ = 'DM.10'

import numpy as np
from numpy.linalg import norm
import utils

maxint = 9223372036854775807

class KMeans:

    def __init__(self, X, K=1, options=None):
        """
         Constructor of KMeans class
             Args:
                 K (int): Number of cluster
                 options (dict): dictºionary with options
            """
        self.centroids = None
        self.old_centroids = None
        self.labels = None
        self.WCD = None
        self.num_iter = 0
        self.K = K
        self._init_X(X)
        self._init_options(options)  # DICT options

    def _init_X(self, x):
        """Initialization of all pixels, sets X as an array of data in vector form (PxD)
            Args:
                x (list or np.array): list(matrix) of all pixel values
                    if matrix has more than 2 dimensions, the dimensionality of the smaple space is the length of
                    the last dimension
        """
        numpy_x = np.array(x)

        # ensures that the values are of the float type;
        if numpy_x.dtype != float:
            numpy_x = numpy_x.astype(float)

        # convert the data to a matrix of 2 dimensions
        if numpy_x.ndim > 2:
            shape = numpy_x.shape
            numpy_x = np.reshape(numpy_x, (shape[0] * shape[1], shape[2]))

        self.X = numpy_x

    def _init_options(self, options=None):
        """
        Initialization of options in case some fields are left undefined
        Args:
            options (dict): dictionary with options
        """
        if options == None:
            options = {}
        if not 'km_init' in options:
            options['km_init'] = 'first'
        if not 'verbose' in options:
            options['verbose'] = False
        if not 'tolerance' in options:
            options['tolerance'] = 0.0
        if not 'max_iter' in options:
            options['max_iter'] = maxint
        if not 'fitting' in options:
            options['fitting'] = 'WCD'  #within class distance.

        # If your methods need any other prameter you can add it to the options dictionary
        self.options = options

        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################

    def _init_centroids(self):
        """
        Initialization of centroids
        """

        if self.options['km_init'].lower() == 'first':
            self.centroids = []
            self.centroids.append(self.X[0])
            initialized_centroids = 1
            for pixel in self.X:
                is_repeated = False
                for centroid in self.centroids:
                    if np.array_equal(pixel, centroid):
                        is_repeated = True
                if not is_repeated:
                    self.centroids.append(pixel)
                    initialized_centroids += 1
                    if initialized_centroids == self.K:
                        break
            self.old_centroids = self.centroids
#        elif self.options['km_init'].lower() == 'custom':

        else:
            self.centroids = np.random.rand(self.K, self.X.shape[1])
            self.old_centroids = np.random.rand(self.K, self.X.shape[1])

    def get_labels(self):
        """        Calculates the closest centroid of all points in X
        and assigns each point to the closest centroid
        """
        labels = []
        for pixel in self.X:
            lower_distance = maxint
            nearest_centroid = -1
            for index in range(len(self.centroids)):
                distance = norm(np.array(pixel) - np.array(self.centroids[index]))
                if distance < lower_distance:
                    lower_distance = distance
                    nearest_centroid = index
            labels.append(nearest_centroid)

        self.labels = labels

    def get_centroids(self):
        """
        Calculates coordinates of centroids based on the coordinates of all the points assigned to the centroid
        """

        self.old_centroids = self.centroids
        new_centroids = [[] for k in range(self.K)]

        for pixel_index in range(len(self.X)):
            new_centroids[self.labels[pixel_index]].append(self.X[pixel_index])

        for pixel_group_index in range(len(new_centroids)):
            new_centroids[pixel_group_index] = np.average(np.array(new_centroids[pixel_group_index]), 0)

        self.centroids = new_centroids

        pass

    def converges(self):
        """
        Checks if there is a difference between current and old centroids
        """
        return np.allclose(self.centroids, self.old_centroids, rtol=self.options['tolerance'],
                           atol=self.options['tolerance'])

    def fit(self):
        """
        Runs K-Means algorithm until it converges or until the number
        of iterations is smaller than the maximum number of iterations.
        """

        self._init_centroids()
        iterations = self.options['max_iter']
        while iterations > 0:
            self.get_labels()
            self.get_centroids()
            if self.converges():
                break
            iterations -= 1

    def within_class_distance(self):
        """
         returns the whithin class distance of the current clustering
        """

        sumatory = 0

        for pixel_index in range(len(self.X)):
            centroid = self.labels[pixel_index]
            pixel = self.X[pixel_index]

            inter_class_distance = norm(np.array(pixel) - np.array(self.centroids[centroid])) ** 2

            sumatory += inter_class_distance

        wcd = (1 / len(self.X)) * sumatory

        self.WCD = wcd

    def find_bestK(self, max_K):
        """
         sets the best k anlysing the results up to 'max_K' clusters
        """

        # first iteration on k = 2

        self.K = 2
        self.fit()
        self.within_class_distance()
        last_wcd = self.WCD

        test_k = 3
        threshold = 20

        while test_k <= max_K:

            self.K = test_k
            self.fit()
            self.within_class_distance()

            decrease = 100 * self.WCD / last_wcd

            if 100 - decrease < threshold:
                ideal_k = test_k - 1
                self.K = ideal_k
                break

            last_wcd = self.WCD
            test_k += 1

        pass

def distance(X, C):
    """
    Calculates the distance between each pixel and each centroid
    Args:
        X (numpy array): PxD 1st set of data points (usually data points)
        C (numpy array): KxD 2nd set of data points (usually cluster centroids points)

    Returns:
        dist: PxK numpy array position ij is the distance between the
        i-th point of the first set an the j-th point of the second set
    """

    dist = []
    for pixel in X:
        pixel_distances = []
        for centroid in C:
            pixel_distances.append(norm(np.array(pixel)-np.array(centroid)))
        dist.append(pixel_distances)
    return dist


def get_colors(centroids):
    """
    for each row of the numpy matrix 'centroids' returns the color laber folllowing the 11 basic colors as a LIST
    Args:
        centroids (numpy array): KxD 1st set of data points (usually centroind points)

    Returns:
        lables: list of K labels corresponding to one of the 11 basic colors
    """

    color_probabilities = utils.get_color_prob(centroids)

    color_labels = []
    for centroid_index in range(len(centroids)):
        higher_probability = 0
        color_index = -1
        for probability_index in range(len(color_probabilities[centroid_index])):
            probability = color_probabilities[centroid_index][probability_index]
            if probability > higher_probability:
                higher_probability = probability
                color_index = probability_index
        color_labels.append(utils.colors[color_index])

    return color_labels
