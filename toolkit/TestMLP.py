from __future__ import (absolute_import, division, print_function, unicode_literals)

from unittest import TestCase,TestLoader,TextTestRunner

from toolkit.manager import plot_data
from .Backprop import BackProp
from .MultiPerceptron import MultiPerceptron
from .matrix import Matrix
import time


class TestMLP(TestCase):

    data = Matrix()
    #data.load_arff("datasets/votingMissingValuesReplaced.arff")
    #data.load_arff("datasets/not-linearly-seperable-data.arff")
    data.load_arff("test/testbp.arff")
    learn = BackProp(hidden_layers=2, nodes_per_layer=2, output_nodes=2)

    def test_complete(self):
        features = Matrix(self.data, 0, 0, self.data.rows, self.data.cols - 2)
        labels = Matrix(self.data, 0, self.data.cols - 2, self.data.rows, self.data.cols)
        confusion = Matrix()
        start_time = time.time()
        self.learn.train(features, labels)
        elapsed_time = time.time() - start_time
        print("Time to train (in seconds): {}".format(elapsed_time))
        accuracy = self.learn.measure_accuracy(features, labels, confusion)
        print("Training set accuracy: " + str(accuracy))
        print("Weights for the Perceptron are: {}".format(self.learn.weights))
        print("Accuracy is: {}".format(self.learn.accuracy_hash))
        plot_data(features.data, labels.data, self.learn.weights)
        # plot the answers
        import matplotlib.pylab as plt
        mis = {key: 1 - self.learn.accuracy_hash[key] for key in self.learn.accuracy_hash.keys()}
        lists = sorted(mis.items())  # sorted by key, return a list of tuples
        x, y = zip(*lists)  # unpack a list of pairs into two tuples
        plt.plot(x, y)
        plt.xlabel('Epoch')
        plt.ylabel('Misclassification Rate')
        plt.title('Training Misclassification')
        axes = plt.gca()
        axes.set_ylim([0, 1])
        plt.show()

        print('#### done ####')