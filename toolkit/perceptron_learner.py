from __future__ import (absolute_import, division, print_function, unicode_literals)

from supervised_learner import SupervisedLearner

import numpy  as np

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
        end = True
        i = 0
        old_w = np.copy(self.w)
        while end:
            self.w = self.epoch(self.c, F, np.squeeze(np.asarray(l.data)), self.w, Feat.rows)
            if np.mod(i, 10) == 0:
                if np.sum(weight(old_w)) - np.sum(weight(self.w)) < .0001:
                    end = False
                    print("Epochs:", i)
                    # print("Weights:")
                    # print(self.w[0])
                    # print(self.w[1])
                    # print(self.w[2])
                    # print(self.w[3])
                    # print(self.w[4])
                    # print(self.w[5])
                    # print(self.w[6])
                    # print(self.w[7])
                    # print(self.w[8])
                    # print(self.w[9])
                    # print(self.w[10])
                    # print(self.w[11])
                    # print(self.w[12])
                    # print(self.w[13])
                    # print(self.w[14])
                    # print(self.w[15])

                old_w = np.copy(self.w)
            i += 1
        return self.w


    def predict(self, f, l): # test
        """
        :type f: [float, ...]
        :type l: [float]
        """
        f.append(1)
        del l[:]
        l += [(self.test(f, self.w))]



