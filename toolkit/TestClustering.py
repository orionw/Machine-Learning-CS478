from __future__ import (absolute_import, division, print_function, unicode_literals)

from unittest import TestCase,TestLoader,TextTestRunner

from toolkit.manager import plot_data
from .Clustering import Cluster
from .matrix import Matrix
import time
import numpy as np
from scipy.interpolate import splrep, splev, spline



class TestDecisionTree(TestCase):


    data = Matrix()
    data.load_arff("test/laborWithId.arff")
    # data.load_arff("datasets/sponge.arff")

    #test.load_arff("test/testknnhw.arff")
    #train.load_arff("test/trainknnhw.arff")
    # test.load_arff("datasets/mt_test.arff")
    # train.load_arff("datasets/mt_train.arff")
    # #
    # test.load_arff("datasets/housing-test.arff")
    # train.load_arff("datasets/housing-train.arff")
    # test.load_arff("datasets/knn_train.arff")
    # train.load_arff("datasets/knn_test.arff")

    def test_complete(self):
        self.learn = Cluster(type="K-Means", k=5, link_type="complete")
        # # 75/25 split
        # self.test.shuffle()
        # self.train.shuffle()

        features = Matrix(self.data, 0, 1, self.data.rows, self.data.cols - 2)
        labels = Matrix(self.data, 0, self.data.cols - 1, self.data.rows, self.data.cols)
        features.normalize()
        start_time = time.time()
        self.learn.train(features, labels)
        elapsed_time = time.time() - start_time
        print("Time to train (in seconds): {}".format(elapsed_time))
