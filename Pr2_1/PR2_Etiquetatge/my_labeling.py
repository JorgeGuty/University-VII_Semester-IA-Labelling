__authors__ = 'TO_BE_FILLED'
__group__ = 'TO_BE_FILLED'

import numpy as np
from Kmeans import *
from KNN import *

from utils_data import read_dataset, visualize_k_means, visualize_retrieval
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import cv
import threading
from time import process_time


# You can start coding your functions here

def retrieval_combined(images_list, shape_tags, color_tags, shape_question, color_question):
    # get the images that match both

    matches = []
    for image_index in range(len(images_list)):
        shape_tag = shape_tags[image_index]
        color_tag = color_tags[image_index]
        print()
        if shape_tag == shape_question and set(color_tag + color_question) == set(color_tag):
            matches.append(images_list[image_index])

    return matches


def k_mean_statistics(loaded_kmeans_list, maxk):
    wcd_lists = [[] for _ in range(len(loaded_kmeans_list))]
    time_lists = [[] for _ in range(len(loaded_kmeans_list))]
    tested_k_list = list(range(2, maxk+1))
    kmean_fit_threads = []
    print("initializing threads...")
    for kmean_index in range(len(loaded_kmeans_list)):
        kmean_thread = threading.Thread(target=fit_til_maxk,
                                        args=(loaded_kmeans_list[kmean_index],
                                              maxk,
                                              wcd_lists[kmean_index],
                                              time_lists[kmean_index]))
        kmean_fit_threads.append(kmean_thread)
        kmean_thread.start()
    print("all threads working...")

    for thread in kmean_fit_threads:
        thread.join()
        print("thread joined")

    wcd_averages = np.average(wcd_lists, axis=0)[0]
    time_averages = np.average(time_lists, axis=0)[0]

    # labels the x axis with Months
    plt.xlabel('K values')

    # labels the y axis with Number Potatoes
    plt.ylabel('WCD average in the test images')

    # changes the title of our graph
    plt.title('WCD averages with different k values.')
    print(tested_k_list, wcd_averages)
    plt.plot(tested_k_list, wcd_averages, color='b')
    plt.show()

    # labels the x axis with Months
    plt.xlabel('K values')

    # labels the y axis with Number Potatoes
    plt.ylabel('Computation time in seconds')

    plt.plot(tested_k_list, time_averages, color='r')
    plt.show()

    pass


def fit_til_maxk(kmeans, maxk, wcd_results, time_results):
    test_k = 2
    wcd_list = []
    time_list = []
    while test_k <= maxk:
        print(test_k)
        kmeans.K = test_k
        start = process_time()
        kmeans.fit()
        finish = process_time()
        elapsed = finish - start
        time_list.append(elapsed)
        kmeans.within_class_distance()
        wcd_list.append(kmeans.WCD)
        test_k += 1
    wcd_results.append(wcd_list)
    time_results.append(time_list)
    print("Finished til maxk")

# Load all the images and GT
train_imgs, train_class_labels, train_color_labels, \
test_imgs, test_class_labels, test_color_labels = read_dataset(ROOT_FOLDER='./images/', gt_json='./images/gt.json')

# List with all the existant classes
classes = list(set(list(train_class_labels) + list(test_class_labels)))

# test of retrieval_combined
# test_imgs = test_imgs[:40]
# knn = KNN(train_imgs, train_class_labels)
# color_results = []
# label_results = knn.predict(test_imgs, 10)
# for image in test_imgs:
#     km = KMeans(image, 3)
#     km.fit()
#     colors = get_colors(np.array([list(km.centroids[0]), list(km.centroids[1]), list(km.centroids[2])]))
#     color_results.append(colors)
#
# print(color_results)
# print(label_results)

# # test 1: grey flip flops
# print("Retrieving grey flip-flops")
# grey_flip_flops = retrieval_combined(test_imgs, label_results, color_results, "Flip Flops", ["Grey"])
# for image in grey_flip_flops:
#     imageObj = Image.fromarray(image)
#     imageObj.show()
#
# #test two: brown dresses
# print("Retrieving brown dresses")
# brown_dresses = retrieval_combined(test_imgs, label_results, color_results, "Dresses", ["Brown"])
# for image in brown_dresses:
#     imageObj = Image.fromarray(image)
#     imageObj.show()
#
# #test two: red socks
# print("Retrieving red socks")
# red_socks = retrieval_combined(test_imgs, label_results, color_results, "Socks", ["Red"])
# for image in red_socks:
#     imageObj = Image.fromarray(image)
#     imageObj.show()

# test of k_mean_statistics

kmeans_list = []
for image in test_imgs[:10]:
    kmeans_instance = KMeans(image)
    kmeans_list.append(kmeans_instance)

k_mean_statistics(kmeans_list, 5)

options = {"km_init": "kmeans++"}
kmeans_list = []
for image in test_imgs[:10]:
    kmeans_instance = KMeans(image)
    kmeans_list.append(kmeans_instance)

k_mean_statistics(kmeans_list, 5)

options = {"km_init": "naive"}
kmeans_list = []
for image in test_imgs[:10]:
    kmeans_instance = KMeans(image)
    kmeans_list.append(kmeans_instance)

k_mean_statistics(kmeans_list, 5)



