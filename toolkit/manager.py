from __future__ import (absolute_import, division, print_function, unicode_literals)

from toolkit.supervised_learner import SupervisedLearner
from toolkit.baseline_learner import BaselineLearner
from toolkit.perceptron import Perceptron
# from toolkit.multi_perceptron_learner import MultiPerceptronLearner
# from toolkit.backprop_learner import BackpropLearner
# from toolkit.decision_tree_learner import DecisionTreeLearner
# from toolkit.k_means_cluster_learner import KMeansClusterLearner
# from toolkit.hac_cluster_learner import HACClusterLearner
from toolkit.instance_based_learner import InstanceBasedLearner
from toolkit.matrix import Matrix
import random
import argparse
import time
import numpy as np
import matplotlib.pyplot  as plt


class MLSystemManager:
    def get_learner(self, model):
        """
        Get an instance of a learner for the given model name.

        To use toolkitPython as external package, you can extend this class (MLSystemManager)
        with your own custom class located outside of this package, and override this method
        to return your custom learners.

        :type model: str
        :rtype: SupervisedLearner
        """
        modelmap = {
            "baseline": BaselineLearner(),
            "perceptron": Perceptron(),
            # "multiperceptron": MultiPerceptronLearner(),
            # "backprop": BackpropLearner(),
            # "decisiontree": DecisionTreeLearner(),
            "knn": InstanceBasedLearner(),
            # "k_means": KMeansClusterLearner(),
            # "hac": HACClusterLearner()
        }
        if model in modelmap:
            return modelmap[model]
        else:
            raise Exception("Unrecognized model: {}".format(model))

    def main(self):
        # parse the command-line arguments
        args = self.parser().parse_args()
        file_name = args.arff
        learner_name = args.L
        eval_method = args.E[0]
        eval_parameter = args.E[1] if len(args.E) > 1 else None
        print_confusion_matrix = args.verbose
        normalize = args.normalize
        random.seed(args.seed)  # Use a seed for deterministic results, if provided (makes debugging easier)

        # load the model
        learner = self.get_learner(learner_name)

        # load the ARFF file
        data = Matrix()
        data.load_arff(file_name)
        if normalize:
            print("Using normalized data")
            data.normalize()

        # print some stats
        print("\nDataset name: {}\n"
              "Number of instances: {}\n"
              "Number of attributes: {}\n"
              "Learning algorithm: {}\n"
              "Evaluation method: {}\n".format(file_name, data.rows, data.cols, learner_name, eval_method))

        if eval_method == "training":

            print("Calculating accuracy on training set...")

            features = Matrix(data, 0, 0, data.rows, data.cols - 1)  # todo: edit here
            labels = Matrix(data, 0, data.cols - 1, data.rows, data.cols)
            confusion = Matrix()
            start_time = time.time()
            learner.train(features, labels)
            elapsed_time = time.time() - start_time
            print("Time to train (in seconds): {}".format(elapsed_time))
            accuracy = learner.measure_accuracy(features, labels, confusion)
            print("Training set accuracy: " + str(accuracy))

            if print_confusion_matrix:
                print("\nConfusion matrix: (Row=target value, Col=predicted value)")
                confusion.print()
                print("")

        elif eval_method == "static":

            print("Calculating accuracy on separate test set...")

            test_data = Matrix(arff=eval_parameter)
            if normalize:
                test_data.normalize()

            print("Test set name: {}".format(eval_parameter))
            print("Number of test instances: {}".format(test_data.rows))
            features = Matrix(data, 0, 0, data.rows, data.cols - 1)
            labels = Matrix(data, 0, data.cols - 1, data.rows, 1)

            start_time = time.time()
            learner.train(features, labels)
            elapsed_time = time.time() - start_time
            print("Time to train (in seconds): {}".format(elapsed_time))

            train_accuracy = learner.measure_accuracy(features, labels)
            print("Training set accuracy: {}".format(train_accuracy))

            test_features = Matrix(test_data, 0, 0, test_data.rows, test_data.cols - 1)
            test_labels = Matrix(test_data, 0, test_data.cols - 1, test_data.rows, 1)
            confusion = Matrix()
            test_accuracy = learner.measure_accuracy(test_features, test_labels, confusion)

            plot_data(features.data, labels.data, learner.__getattribute__("weights") )

            print("Test set accuracy: {}".format(test_accuracy))

            if print_confusion_matrix:
                print("\nConfusion matrix: (Row=target value, Col=predicted value)")
                confusion.print()
                print("")

        elif eval_method == "random":

            print("Calculating accuracy on a random hold-out set...")
            train_percent = float(eval_parameter)
            if train_percent < 0 or train_percent > 1:
                raise Exception("Percentage for random evaluation must be between 0 and 1")
            print("Percentage used for training: {}".format(train_percent))
            print("Percentage used for testing: {}".format(1 - train_percent))

            data.shuffle()

            train_size = int(train_percent * data.rows)
            train_features = Matrix(data, 0, 0, train_size, data.cols - 1)
            train_labels = Matrix(data, 0, data.cols - 1, train_size, 1)

            test_features = Matrix(data, train_size, 0, data.rows - train_size, data.cols - 1)
            test_labels = Matrix(data, train_size, data.cols - 1, data.rows - train_size, 1)

            start_time = time.time()
            learner.train(train_features, train_labels)
            elapsed_time = time.time() - start_time
            print("Time to train (in seconds): {}".format(elapsed_time))

            train_accuracy = learner.measure_accuracy(train_features, train_labels)
            print("Training set accuracy: {}".format(train_accuracy))

            confusion = Matrix()
            test_accuracy = learner.measure_accuracy(test_features, test_labels, confusion)
            print("Test set accuracy: {}".format(test_accuracy))

            if print_confusion_matrix:
                print("\nConfusion matrix: (Row=target value, Col=predicted value)")
                confusion.print()
                print("")

        elif eval_method == "cross":

            print("Calculating accuracy using cross-validation...")

            folds = int(eval_parameter)
            if folds <= 0:
                raise Exception("Number of folds must be greater than 0")
            print("Number of folds: {}".format(folds))
            reps = 1
            sum_accuracy = 0.0
            elapsed_time = 0.0
            for j in range(reps):
                data.shuffle()
                for i in range(folds):
                    begin = int(i * data.rows / folds)
                    end = int((i + 1) * data.rows / folds)

                    train_features = Matrix(data, 0, 0, begin, data.cols - 1)
                    train_labels = Matrix(data, 0, data.cols - 1, begin, 1)

                    test_features = Matrix(data, begin, 0, end - begin, data.cols - 1)
                    test_labels = Matrix(data, begin, data.cols - 1, end - begin, 1)

                    train_features.add(data, end, 0, data.cols - 1)
                    train_labels.add(data, end, data.cols - 1, 1)

                    start_time = time.time()
                    learner.train(train_features, train_labels)
                    elapsed_time += time.time() - start_time

                    accuracy = learner.measure_accuracy(test_features, test_labels)
                    sum_accuracy += accuracy
                    print("Rep={}, Fold={}, Accuracy={}".format(j, i, accuracy))

            elapsed_time /= (reps * folds)
            print("Average time to train (in seconds): {}".format(elapsed_time))
            print("Mean accuracy={}".format(sum_accuracy / (reps * folds)))

        else:
            raise Exception("Unrecognized evaluation method '{}'".format(eval_method))

    def parser(self):
        parser = argparse.ArgumentParser(description='Machine Learning System Manager')

        parser.add_argument('-V', '--verbose', action='store_true', help='Print the confusion matrix and learner '
                                                                         'accuracy on individual class values')
        parser.add_argument('-N', '--normalize', action='store_true', help='Use normalized data')
        parser.add_argument('-R', '--seed', help="Random seed")  # will give a string
        parser.add_argument('-L', required=True,
                            choices=['baseline', 'perceptron', "multiperceptron", "backprop", 'neuralnet',
                                     'decisiontree', 'knn', 'k_means', 'hac'], help='Learning Algorithm')
        parser.add_argument('-A', '--arff', metavar='filename', required=True, help='ARFF file')
        parser.add_argument('-E', metavar=('METHOD', 'args'), required=True, nargs='+',
                            help="Evaluation method (training | static <test_ARFF_file> | random <%%_for_training> "
                                 "| cross <num_folds>)")

        return parser


def plot_data(inputs, targets, weights):
    if len(weights) < 3 or len(weights) > 3:
        return

    # fig config
    inputs = np.asarray(inputs)
    targets = np.asarray(targets)
    weights = np.asarray(weights)

    plt.figure(figsize=(10, 6))
    plt.grid(True)

    # plot input samples(2D data points) and i have two classes.
    # one is +1 and second one is -1, so it red color for +1 and blue color for -1
    for input, target in zip(inputs, targets):
        plt.plot(input[0], input[1], 'ro' if (target == 1.0) else 'bo')

    # Here i am calculating slope and intercept with given three weights
    for i in np.linspace(np.amin(inputs[:, :1]), np.amax(inputs[:, :1])):
        slope = -(weights[0] / weights[2]) / (weights[0] / weights[1])
        intercept = -weights[0] / weights[2]

        # y =mx+c, m is slope and c is intercept
        y = (slope * i) + intercept

        plt.plot(i, y, 'ko')

    plt.show()


if __name__ == '__main__':
    MLSystemManager().main()
