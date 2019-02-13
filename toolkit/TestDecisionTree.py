from __future__ import (absolute_import, division, print_function, unicode_literals)

from unittest import TestCase,TestLoader,TextTestRunner

from toolkit.manager import plot_data
from .DecisionTree import DecisionTree
from .MultiPerceptron import MultiPerceptron
from .matrix import Matrix
import time
import numpy as np
from scipy.interpolate import splrep, splev, spline



class TestDecisionTree(TestCase):

    data = Matrix()
    data.load_arff("test/tennis.arff")
    #data.load_arff("datasets/vowel.arff")
    #data.load_arff("datasets/iris.arff")
    learn = DecisionTree()

    def test_complete(self):
        # 75/25 split
        # train_size = int(.75 * self.data.rows)
        # train_features = Matrix(self.data, 0, 0, train_size, self.data.cols - 1)
        # train_labels = Matrix(self.data, 0, self.data.cols - 1, train_size, 1)
        #
        # test_features = Matrix(self.data, train_size, 0, self.data.rows - train_size, self.data.cols - 1)
        # test_labels = Matrix(self.data, train_size, self.data.cols - 1, self.data.rows - train_size, 1)

        train_features = Matrix(self.data, 0, 0, self.data.rows, self.data.cols - 1)
        train_labels = Matrix(self.data, 0, self.data.cols - 1, self.data.rows, 1)

        start_time = time.time()
        self.learn.train(train_features, train_labels)
        elapsed_time = time.time() - start_time
        print("Time to train (in seconds): {}".format(elapsed_time))

        test_accuracy = self.learn.measure_accuracy(train_features, train_labels, MSE=False)
        print("Test set accuracy: {}".format(test_accuracy))


        # # plot the answers
        # import matplotlib.pylab as plt
        # # validation set misclassification
        # mis = {key: self.learn.accuracy_hash[key] for key in self.learn.accuracy_hash.keys()}
        # lists = sorted(mis.items())  # sorted by key, return a list of tuples
        # x, y = zip(*lists)  # unpack a list of pairs into two tuples
        #
        # # training misclassification
        # mis = {key: self.learn.accuracy_hash_train[key] for key in self.learn.accuracy_hash_train.keys()}
        # lists = sorted(mis.items())  # sorted by key, return a list of tuples
        # x_train, y_train = zip(*lists)  # unpack a list of pairs into two tuples
        #
        #
        # plt.plot(x, y)
        # plt.plot(x_train, y_train)
        # my_xticks = np.array(x)
        # frequency = 20
        # plt.xticks(x[::frequency], my_xticks[::frequency])
        # plt.xlabel('Epoch')
        # plt.ylabel('MSE')
        # plt.title('MSE Across Number of Hidden Nodes')
        # axes = plt.gca()
        # axes.set_ylim([0, 1])
        # axes.legend(['VS MSE', 'Training MSE'])
        # plt.show()
        #
        # print('#### done ####')