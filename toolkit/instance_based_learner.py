from __future__ import (absolute_import, division, print_function, unicode_literals)

from toolkit.supervised_learner import SupervisedLearner
from toolkit.matrix import Matrix
import numpy as np


def o(x, b):
    if np.isinf(x) or np.isnan(x):
        return 1
    if x != 0 and b:
        return 1
    return x

o = np.vectorize(o)

def replace_nan(x, z):
    if np.isnan(x):
        return z
    return x

replace_nan = np.vectorize(replace_nan)

def replace_inf(x, z):
    if np.isinf(x):
        return z
    return x

replace_inf = np.vectorize(replace_inf)

def count_var_with_name(x, name):
    if x == name:
        return 1
    else:
        return 0

count_var_with_name = np.vectorize(count_var_with_name)


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
        self.info = []
        pass

    def calc_info_s(self, data):
        info = 0
        array_size = np.size(data, axis=0)
        for i in range(len(self.is_nominal)):
            # noinspection PyTypeChecker
            num = np.sum(count_var_with_name(data, i), axis=0)[-1]
            p = num / array_size
            if p != 0:
                info -= p * np.log2(p)
        return info

    def calc_info_var(self, data, j):
        info = 0
        r = len(self.is_nominal)
        array_size = np.size(data, axis=0)
        for i in range(r):
            # noinspection PyTypeChecker
            num = np.sum(count_var_with_name(data, i), axis=0)[j]
            p = num / array_size
            if p != 0:
                data_copy = self.idk(data, j, i)
                info += p * self.calc_info_s(data_copy)
        return info

    def idk(self, data, j, value):
        data_new = []
        for line in data:
            if line[j] == value:
                data_new += [line]
        return np.asarray(data_new)

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
            args = np.argmin(features.data, axis=0)
            for i in range(np.size(args)):
                self.min_features += [np.array(features.data)[args[i]][i]]
            args = np.argmax(replace_inf(features.data, float("-inf")), axis=0)
            for i in range(np.size(args)):
                self.max_features += [np.array(features.data)[args[i]][i]]
            self.features = (np.array(features.data) - np.array(self.min_features)) / (np.array(self.max_features) -
                                                                                       np.array(self.min_features))
        else:
            self.features = np.array(features.data)

        data = np.hstack((features.data, labels.data))
        for i in range(np.size(data, axis=1) - 1):
            self.info += [self.calc_info_var(data, i)]
        self.labels = labels.data

    def predict(self, features, labels):
        """
        :type features: [float]
        :type labels: [float]
        """
        if self.normalization:
            features = (np.array(features) - np.array(self.min_features)) / (np.array(self.max_features) -
                                                                             np.array(self.min_features))
        features = o(features, self.is_nominal)

        f = np.array(self.features) - np.array(features)
        f = replace_inf(f, 1)
        f = replace_nan(f, 1)
        f = f * (1 - np.array(self.info))
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

