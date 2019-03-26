from __future__ import (absolute_import, division, print_function, unicode_literals)

import math
from unittest import TestCase,TestLoader,TextTestRunner

from toolkit.manager import plot_data
from .Clustering import Cluster
from .matrix import Matrix
import time
import numpy as np
from scipy.interpolate import splrep, splev, spline



class TestClustering(TestCase):


    data = Matrix()
    # data.load_arff("test/test_hac.arff")
    # data.load_arff("datasets/sponge.arff")
    # data.load_arff("test/laborWithID.arff")


    #test.load_arff("test/testknnhw.arff")
    #train.load_arff("test/trainknnhw.arff")
    # data.load_arff("datasets/iris.arff")
    data.load_arff("datasets/abalone500.arff")
    # train.load_arff("datasets/mt_train.arff")
    # #
    # test.load_arff("datasets/housing-test.arff")
    # train.load_arff("datasets/housing-train.arff")
    # test.load_arff("datasets/knn_train.arff")
    # train.load_arff("datasets/knn_test.arff")

    def test_complete(self):
        list_of_sse = []
        list_of_sil = []
        for i in range(2, 8):
            self.learn = Cluster(type="K-Means", k=i, link_type="single", use_median=True, random=True)
            # # 75/25 split
            # self.test.shuffle()
            # self.train.shuffle()
            # features = self.data
            features = Matrix(self.data, 0, 0, self.data.rows, self.data.cols)
            features.normalize()
            # features = Matrix(self.data, 0, 1, self.data.rows, self.data.cols - 2)
            # labels = Matrix(self.data, 0, self.data.cols - 1, self.data.rows, self.data.cols)
            # features.normalize()
            start_time = time.time()
            self.learn.train(features, features)
            elapsed_time = time.time() - start_time
            list_of_sse.append(self.learn.total_sse)
            list_of_sil.append(np.mean(self.learn.get_silhouette(self.learn.clustering, features.return_pandas_df())))
            print("Time to train (in seconds): {}".format(elapsed_time))
        print(list_of_sse)
        print(list_of_sil)

    def test_silhouette(self):
        data = Matrix()
        data.load_arff("test/test_sil.arff")
        self.learn = Cluster(type="K-Means", k=2)
        start_time = time.time()
        self.learn.train(data, data)
        self.learn.get_silhouette([[0, 1], [2, 3, 4]], data.return_pandas_df(), dist_type="manhattan")
        elapsed_time = time.time() - start_time
        print("Time to train (in seconds): {}".format(elapsed_time))
