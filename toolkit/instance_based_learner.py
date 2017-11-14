from __future__ import (absolute_import, division, print_function, unicode_literals)

from toolkit.supervised_learner import SupervisedLearner
from toolkit.matrix import Matrix
import numpy as np

def o(x, b):
    if x != 0 and b:
        return 1
    return x

o = np.vectorize(o)

class InstanceBasedLearner(SupervisedLearner):
    """
    For nominal labels, this model simply returns the majority class. For
    continuous labels, it returns the mean value.
    If the learning model you're using doesn't do as well as this one,
    it's time to find a new learning model.
    """

    def __init__(self):
        self.classification = False
        self.normalization = True
        self.weighted = True
        self.k = 3

        self.features = []
        self.labels = []
        self.max_features = []
        self.min_features = []
        self.is_nominal = []
        pass

    def train(self, features, labels):
        """
        :type features: Matrix
        :type labels: Matrix
        """
        enum_list = features.enum_to_str
        for enum in enum_list:
            if enum == {}:
                self.is_nominal += [False]
            else:
                self.is_nominal += [True]
        if self.normalization:
            # should do some form of normalization
            args = np.argmax(features.data, axis=0)
            for i in range(np.size(args)):
                self.max_features += [np.array(features.data)[args[i]][i]]
            args = np.argmin(features.data, axis=0)
            for i in range(np.size(args)):
                self.min_features += [np.array(features.data)[args[i]][i]]
            self.features = (np.array(features.data) - np.array(self.min_features)) / (np.array(self.max_features) -
                                                                                       np.array(self.min_features))
        else:
            self.features = np.array(features.data)
        self.labels = labels.data

    def predict(self, features, labels):
        """
        :type features: [float]
        :type labels: [float]
        """

        for i in range(np.size(features, axis=0)):
            if self.is_nominal[i]:
                pass

        if self.normalization:
            features = (np.array(features) - np.array(self.min_features)) / (np.array(self.max_features) -
                                                                             np.array(self.min_features))
        features = o(features, self.is_nominal)

        f = np.array(self.features) - np.array(features)  # todo: figure out what to do with unknown data
        f = np.sum(np.square(f), axis=1)
        k_v = np.argpartition(f, self.k)[:self.k]
        l = 0
        if self.weighted:
            dem = 0
            i = 0
            for index in k_v:
                if f[index] == 0:
                    num = 1 / .0000000000001
                else:
                    num = 1 / f[index]
                l += self.labels[index][0] * num
                dem += num
                i += 1
            l = l / dem
        else:
            for index in k_v:
                l += self.labels[index][0] / self.k
        if self.classification:
            l = np.round(l)
        del labels[:]
        labels += [l]

        # todo: for going farther try to weight the features according to information gain
