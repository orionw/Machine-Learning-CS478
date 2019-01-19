from __future__ import (absolute_import, division, print_function, unicode_literals)

from unittest import TestCase,TestLoader,TextTestRunner

from toolkit.manager import plot_data
from .perceptron import Perceptron
from .MultiPerceptron import MultiPerceptron
from .matrix import Matrix
import time


class TestPerceptronLearner(TestCase):

    data = Matrix()
    data.load_arff("datasets/votingMissingValuesReplaced.arff")
    #self.data.load_arff("datasets/not-linearly-seperable-self.data.arff")
    #self.data.load_arff("datasets/linearly-seperable-self.data.arff")
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
        print("Accuracy is: {}".format(self.learn.accuracy_hash))
        plot_data(features.self.data, labels.self.data, self.learn.weights)
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
        
    def test_random(self):
        # 70/30 split
        train_size = int(.7 * self.data.rows)
        train_features = Matrix(self.data, 0, 0, train_size, self.data.cols - 1)
        train_labels = Matrix(self.data, 0, self.data.cols - 1, train_size, 1)

        test_features = Matrix(self.data, train_size, 0, self.data.rows - train_size, self.data.cols - 1)
        test_labels = Matrix(self.data, train_size, self.data.cols - 1, self.data.rows - train_size, 1)

        start_time = time.time()
        self.learn.train(train_features, train_labels)
        elapsed_time = time.time() - start_time
        print("Time to train (in seconds): {}".format(elapsed_time))

        train_accuracy = self.learn.measure_accuracy(train_features, train_labels)
        print("Training set accuracy: {}".format(train_accuracy))

        confusion = Matrix()
        test_accuracy = self.learn.measure_accuracy(test_features, test_labels, confusion)
        print("Test set accuracy: {}".format(test_accuracy))

        print("Epochs are: {}".format(self.learn.epoch_count))

        print("Weights for the Perceptron are: {}".format(self.learn.weights))
        print("Accuracy is: {}".format(self.learn.accuracy_hash))
        # plot the answers
        import matplotlib.pylab as plt
        mis = {key: 1 - self.learn.accuracy_hash[key] for key in self.learn.accuracy_hash.keys()}
        lists = sorted(mis.items())  # sorted by key, return a list of tuples
        x, y = zip(*lists)  # unpack a list of pairs into two tuples
        plt.plot(x, y)
        plt.xlabel('Epoch')
        plt.ylabel('Avg. Misclassification Rate')
        plt.title('Avg Misclassification Rate over Epochs')
        axes = plt.gca()
        axes.set_ylim([0, 1])
        plt.show()
        

    def test_multi(self):
        self.data = Matrix()
        self.data.load_arff("datasets/iris_small.arff")
        self.learn = MultiPerceptron()
        features = Matrix(self.data, 0, 0, self.data.rows, self.data.cols - 1)
        labels = Matrix(self.data, 0, self.data.cols - 1, self.data.rows, 1)
        confusion = Matrix()
        start_time = time.time()
        self.learn.train(features, labels)
        elapsed_time = time.time() - start_time
        print("Time to train (in seconds): {}".format(elapsed_time))
        accuracy = self.learn.measure_accuracy(features, labels, confusion)
        print("Training set accuracy: " + str(accuracy))
        print(confusion.print())



        print('#### done ####')


# suite = TestLoader().loadTestsFromTestCase(TestPerceptronLearner)
# TextTestRunner(verbosity=2).run(suite)