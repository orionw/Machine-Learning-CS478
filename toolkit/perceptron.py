from __future__ import (absolute_import, division, print_function, unicode_literals)

from toolkit.supervised_learner import SupervisedLearner
from toolkit.matrix import Matrix
import numpy as np


class Perceptron(SupervisedLearner):
    """
    A standard Perceptron implementation.
    """

    labels = []

    def __init__(self):
        self.weights = []
        self.accuracy_hash = {}
        self.epoch_count = 0
        self.learning_rate: float = .1
        self.labels = []

    def _perform_epoch(self, features, labels):
        for row_num in range(features.rows):
            row = features.row(row_num)
            # if len(row) < features.cols + 1:
            #     # add bias - always one
            #     row.append(1)
            # use NumPy for speed
            row = np.array(row)

            output_class = labels.row(row_num)[0]
            net_output = np.dot(row, self.weights)
            pred_class = self._activation(net_output)
            # update weights
            if output_class - pred_class != 0:
                constants = self.learning_rate * (output_class - pred_class)
                self.weights = constants * row + self.weights

        return self.weights


    def _accuracy_increasing(self, margin_increasing):
        # print(self.accuracy_hash)
        accuracy_increasing = False
        baseline: float = self.accuracy_hash[str(self.epoch_count - 20)]
        # check the end point also
        for iteration in range(self.epoch_count - 19, self.epoch_count + 1):
            # print(self.accuracy_hash[str(iteration)])
            if self.accuracy_hash[str(iteration)] - baseline > margin_increasing:
                accuracy_increasing = True

        return accuracy_increasing


    def train(self, features, labels):
        """
        :type features: Matrix
        :type labels: Matrix
        """
        # dont forget the bias - it's added at the end
        self.weights = np.zeros(features.cols + 1)
        for row_num in range(features.rows):
            row = features.row(row_num)
            if len(row) < features.cols + 1:
                # add bias - always one
                row.append(1)
        margin_of_increase = .01
        accuracy_increasing = True
        while accuracy_increasing:
            self.epoch_count += 1
            print(" #### On Epoch Number {} with weights {}".format(self.epoch_count, self.weights))
            # train on the whole training set
            self.weights = self._perform_epoch(features, labels)
            # get accuracy for the training set
            accuracy_for_epoch: float = self._measure_accuracy(self.predict(features, labels), labels)
            self.accuracy_hash[str(self.epoch_count)] = accuracy_for_epoch
            # see if perceptron is still learning
            if self.epoch_count > 20:
                accuracy_increasing = self._accuracy_increasing(margin_of_increase)


        # for i in range(labels.cols):
        #     if labels.value_count(i) == 0:
        #         self.labels += [labels.column_mean(i)]          # continuous
        #     else:
        #         self.labels += [labels.most_common_value(i)]    # nominal


    def predict(self, features, labels):
        """
        :type features: [float]
        :type labels: [float]
        """
        if type(features) == Matrix:
            features_data = np.asarray(features.data)
        else:
            features_data = np.asarray(features)
        output_labels = (np.dot(features_data, self.weights))
        if not type(output_labels) == np.ndarray:
            output_labels = np.asarray([(output_labels)])
        output_labels = list(np.where(output_labels > 0, 1, 0))
        return output_labels

    def _activation(self, net_output):
        if net_output > 0:
            return 1
        else:
            return 0

    def _measure_accuracy(self, preds, labels):
        # print(preds)
        # print(labels.data)
        labels = np.asarray(labels.data)
        preds = np.asarray(preds)

        assert(len(preds) == len(labels))
        correct: int = 0
        for i in range(len(preds)):
            if preds[i] == labels[i]:
                correct += 1
        print("Accuracy is {}% with weights {}".format(correct / len(preds), self.weights))
        return correct / len(preds)

