from __future__ import (absolute_import, division, print_function, unicode_literals)

import numpy as np

from toolkit.perceptron_learner import PerceptronLearner

def two_to_zero(x):
    return 1 if x != 2 else 0

two_to_zero = np.vectorize(two_to_zero)

def two_to_one(x):
    return 1 if x == 2 else 0

two_to_one = np.vectorize(two_to_one)

class MultiPerceptronLearner(PerceptronLearner):
    """
    For nominal labels, this model simply returns the majority class. For
    continuous labels, it returns the mean value.
    If the learning model you're using doesn't do as well as this one,
    it's time to find a new learning model.
    """


    def __init__(self):
        self.w = []
        self.c = .1
        self.num_epochs = 1000
        self.W1 = []
        self.W2 = []
        self.W3 = []


    def train(self, F, l):
        """
        :type F : features Matrix
        :type l : labels Matrix
        """
        F_data = np.copy(F.data)
        l_data = np.copy(l.data)
        O_data = np.hstack((F.data, l.data))

        data1 = np.atleast_2d([s for s in O_data if s[F.cols] != 2])
        data2 = np.atleast_2d([s for s in O_data if s[F.cols] != 1])
        data3 = np.atleast_2d([s for s in O_data if s[F.cols] != 0])

        # test class 0 vs class 1
        F.data = np.delete(data1, np.s_[-1:], axis=1)
        l.data = np.delete(data1, np.s_[:4], axis=1)
        self.W1 = PerceptronLearner.train(self, F, l)

        # test class 0 vs class 2 (1)
        F.data = np.delete(data2, np.s_[-1:], axis=1)
        l.data = np.delete(data2, np.s_[:4], axis=1)
        l.data = two_to_one(l.data)
        self.W2 = PerceptronLearner.train(self, F, l)

        # test class 1 vs class 2 (0)
        F.data = np.delete(data3, np.s_[-1:], axis=1)
        l.data = np.delete(data3, np.s_[:4], axis=1)
        l.data = two_to_zero(l.data)
        self.W3 = PerceptronLearner.train(self, F, l)

        F.data = F_data
        l.data = l_data


    def predict(self, f, l): # test
        """
        :type f: [float, ...]
        :type l: [float]
        """
        f = np.append(f, 1)
        del l[:]
        l1 = self.test(f, self.W1)
        l2 = self.test(f, self.W2)
        l3 = self.test(f, self.W3)

        if l1 == 0 and l2 == 0:
            l += [0]
        elif l1 == 1 and l3 == 1:
            l += [1]
        elif l2 == 1 and l3 == 0:
            l += [2]
        l += [1]





