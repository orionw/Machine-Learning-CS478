from __future__ import (absolute_import, division, print_function, unicode_literals)

from unittest import TestCase,TestLoader,TextTestRunner

from toolkit.manager import plot_data
from .perceptron import Perceptron
from .matrix import Matrix
import time


class TestPerceptronLearner(TestCase):

    data = Matrix()
    data.load_arff("datasets/linearly-seperable-data.arff")
    learn = Perceptron()

    def test_complete(self):
        features = Matrix(self.data, 0, 0, self.data.rows, self.data.cols - 1)  # todo: edit here
        labels = Matrix(self.data, 0, self.data.cols - 1, self.data.rows, 1)
        confusion = Matrix()
        start_time = time.time()
        self.learn.train(features, labels)
        elapsed_time = time.time() - start_time
        print("Time to train (in seconds): {}".format(elapsed_time))
        accuracy = self.learn.measure_accuracy(features, labels, confusion)
        print("Training set accuracy: " + str(accuracy))
        print("Weights for the Perceptron are: {}".format(self.learn.weights))
        plot_data(features.data, labels.data, self.learn.weights)
        print('#### done ####')


# suite = TestLoader().loadTestsFromTestCase(TestPerceptronLearner)
# TextTestRunner(verbosity=2).run(suite)