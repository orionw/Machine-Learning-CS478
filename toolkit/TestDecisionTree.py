from __future__ import (absolute_import, division, print_function, unicode_literals)

from unittest import TestCase,TestLoader,TextTestRunner

from toolkit.manager import plot_data
from .DecisionTree import DecisionTree
from .RandomForest import RandomForest
from .matrix import Matrix
import time
import numpy as np
from scipy.interpolate import splrep, splev, spline



class TestDecisionTree(TestCase):

    data = Matrix()
    #data.load_arff("test/tennis.arff")
    #data.load_arff("test/decision_tree_test_2.arff")
    #data.load_arff("test/decision_tree_test_3.arff")
    data.load_arff("datasets/vote.arff")
    #data.load_arff("datasets/cars.arff")


    def test_complete(self):
        self.learn = DecisionTree(pruning=False)

        # 75/25 split
        self.data.shuffle()
        train_size = int(.75 * self.data.rows)
        train_features = Matrix(self.data, 0, 0, train_size, self.data.cols - 1)
        train_labels = Matrix(self.data, 0, self.data.cols - 1, train_size, 1)

        test_features = Matrix(self.data, train_size, 0, self.data.rows - train_size, self.data.cols - 1)
        test_labels = Matrix(self.data, train_size, self.data.cols - 1, self.data.rows - train_size, 1)

        # train_features = Matrix(self.data, 0, 0, self.data.rows, self.data.cols - 1)
        # train_labels = Matrix(self.data, 0, self.data.cols - 1, self.data.rows, 1)


        start_time = time.time()
        self.learn.train(train_features, train_labels)
        elapsed_time = time.time() - start_time
        print("Time to train (in seconds): {}".format(elapsed_time))
        # test_accuracy = self.learn.measure_accuracy(train_features, train_labels, MSE=False)

        test_accuracy = self.learn.measure_accuracy(test_features, test_labels, MSE=False)
        print("Test set accuracy: {}".format(test_accuracy))

    def test_random_forest(self):
        self.learn = RandomForest()
        # 75/25 split
        self.data.shuffle()
        train_size = int(.75 * self.data.rows)
        train_features = Matrix(self.data, 0, 0, train_size, self.data.cols - 1)
        train_labels = Matrix(self.data, 0, self.data.cols - 1, train_size, 1)

        test_features = Matrix(self.data, train_size, 0, self.data.rows - train_size, self.data.cols - 1)
        test_labels = Matrix(self.data, train_size, self.data.cols - 1, self.data.rows - train_size, 1)

        # train_features = Matrix(self.data, 0, 0, self.data.rows, self.data.cols - 1)
        # train_labels = Matrix(self.data, 0, self.data.cols - 1, self.data.rows, 1)

        start_time = time.time()
        self.learn.train(train_features, train_labels)
        elapsed_time = time.time() - start_time
        print("Time to train (in seconds): {}".format(elapsed_time))
        # test_accuracy = self.learn.measure_accuracy(train_features, train_labels, MSE=False)

        test_accuracy = self.learn.measure_accuracy(test_features, test_labels, MSE=False)
        print("Test set accuracy: {}".format(test_accuracy))


