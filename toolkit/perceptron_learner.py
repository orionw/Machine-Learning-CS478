from __future__ import (absolute_import, division, print_function, unicode_literals)

from toolkit.supervised_learner import SupervisedLearner

import numpy as np

def weight(x):
    return np.log(np.abs(x))

weight = np.vectorize(weight)

class PerceptronLearner(SupervisedLearner):
    """
    For nominal labels, this model simply returns the majority class. For
    continuous labels, it returns the mean value.
    If the learning model you're using doesn't do as well as this one,
    it's time to find a new learning model.
    """

    labels = []

    def __init__(self):
        self.w = []
        self.c = .1
        self.num_epochs = 1000
        pass

    def f(self, x):
        return 1 if x > 0 else 0

    # f = np.vectorize(f)

    def test(self, F, w):
        n = np.dot(F, w)
        z = self.f(n)
        return z

    # c = learning scalar, f= features vector, t = target vector, w = weight vector
    def delta_w(self, c, F, t, w):
        n = np.dot(F, w)
        z = self.f(n)
        delta = (c * (t - z)) * F
        return w + delta

    # c = learning scalar, f= features vector, t = target vector, w = weight vector
    def epoch(self, c, F, t, w, N):
        for i in range(N):
            w = self.delta_w(c, F[i], t[i], w)
        return w


    def train(self, Feat, l):
        """
        :type F : features Matrix
        :type l : labels Matrix
        """
        self.w = np.random.randn(Feat.cols + 1)
        F = np.squeeze(np.asarray(Feat.data))
        F = np.hstack((F, np.ones((Feat.rows, 1))))
        for _ in range(self.num_epochs):
            self.w = self.epoch(self.c, F, np.squeeze(np.asarray(l.data)), self.w, Feat.rows)
        return self.w


    def predict(self, f, l):  # test
        """
        :type f: [float, ...]
        :type l: [float]
        """
        f.append(1)
        del l[:]
        l += [(self.test(f, self.w))]



