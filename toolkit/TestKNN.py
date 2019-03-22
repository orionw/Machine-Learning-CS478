from __future__ import (absolute_import, division, print_function, unicode_literals)

from unittest import TestCase,TestLoader,TextTestRunner

from toolkit.manager import plot_data
from .KNN import KNN
from .matrix import Matrix
import time
import numpy as np
from scipy.interpolate import splrep, splev, spline



class TestDecisionTree(TestCase):

    test = Matrix()
    train = Matrix()
    data = Matrix()
    data.load_arff("datasets/credit-a.arff")
    #test.load_arff("test/testknnhw.arff")
    #train.load_arff("test/trainknnhw.arff")
    test.load_arff("datasets/mt_test.arff")
    train.load_arff("datasets/mt_train.arff")
    # #
    # test.load_arff("datasets/housing-test.arff")
    # train.load_arff("datasets/housing-train.arff")
    # test.load_arff("datasets/knn_train.arff")
    # train.load_arff("datasets/knn_test.arff")

    def test_complete(self):
        list_values = []
        for type_val in [False, False]:
            print("Weights are {}".format(type_val))
            for i in [3]:
                self.learn = KNN(k=i, weight_distance=type_val, distance_metric="euclidean", use_perceptron=False)

                # # 75/25 split
                # self.test.shuffle()
                # self.train.shuffle()

                #### SPLIT THE DATA OURSELVES ####
                # self.data.shuffle()
                # train_size = int(.85 * self.data.rows)
                # train_features = Matrix(self.data, 0, 0, train_size, self.data.cols - 1)
                # train_labels = Matrix(self.data, 0, self.data.cols - 1, train_size, 1)
                #
                # test_features = Matrix(self.data, train_size, 0, self.data.rows - train_size, self.data.cols - 1)
                # test_labels = Matrix(self.data, train_size, self.data.cols - 1, self.data.rows - train_size, 1)

                # For preset train and test ####
                train_features = Matrix(self.train, 0, 0, self.train.rows, self.train.cols - 1)
                train_labels = Matrix(self.train, 0, self.train.cols - 1, self.train.rows, 1)

                test_features = Matrix(self.test, 0, 0, self.test.rows, self.test.cols - 1)
                test_labels = Matrix(self.test, 0, self.test.cols - 1, self.test.rows, 1)

                # train_features.normalize()
                # test_features.normalize()
                # self.data.normalize()
                # self.data.shuffle()

                start_time = time.time()
                self.learn.train(train_features, train_labels)
                elapsed_time = time.time() - start_time
                print("Time to train (in seconds): {}".format(elapsed_time))
                # test_accuracy = self.learn.measure_accuracy(features, labels, MSE=False)

                test_accuracy = self.learn.measure_accuracy(test_features, test_labels, MSE=False)
                list_values.append(test_accuracy)
                print("Test set accuracy: {}".format(test_accuracy))
            print(list_values)
