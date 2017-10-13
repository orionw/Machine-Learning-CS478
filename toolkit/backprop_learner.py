from __future__ import (absolute_import, division, print_function, unicode_literals)

from toolkit.supervised_learner import SupervisedLearner

import numpy as np


def o(net):
    return 1 / (1 + np.exp(-net))

o = np.vectorize(o)

def d(s):
    return s * (1 - s)

d = np.vectorize(d)


def refactor(l0):
    if l0[0] == 0.0:
        return [1, 0, 0]
    elif l0[0] == 1:
        return [0, 1, 0]
    else:
        return [0, 0, 1]
    pass


class BackpropLearner(SupervisedLearner):
    """
    For nominal labels, this model simply returns the majority class. For
    continuous labels, it returns the mean value.
    If the learning model you're using doesn't do as well as this one,
    it's time to find a new learning model.
    """
    def __init__(self):
        self.w0 = None
        self.w1 = None
        pass

    def train_iteration(self, x, y, out_size=3, len_hidden_layer=5, learning_rate=.01):
        # add the bias
        x = np.append(x, 1).reshape((np.size(x, axis=0) + 1, 1))
        if self.w0 is None:
            self.w0 = np.random.randn(len_hidden_layer, np.size(x, axis=0))
        net0 = np.matmul(self.w0, x)
        op0 = np.append(o(net0), 1).reshape((len_hidden_layer + 1, 1))

        if self.w1 is None:
            self.w1 = np.random.randn(out_size, np.size(op0, axis=0))
        net1 = np.matmul(self.w1, op0)
        op1 = o(net1).reshape((out_size, ))

        # calc determinants
        z = np.argmax(op1)
        z = refactor([z])
        d1 = (y - z) * op1 * (1 - op1)
        tempO0 = np.repeat(op0.reshape((1, len_hidden_layer+1)), out_size, axis=0)
        tempd1 = np.repeat(d1, len_hidden_layer + 1, axis=0).reshape((out_size, len_hidden_layer+1))
        dw1 = np.multiply(np.multiply(tempO0, tempd1), learning_rate)

        d0 = (np.sum(np.multiply(dw1, tempd1), axis=0) * np.transpose(op0) * (1 - np.transpose(op0)))[0][:-1]
        tempx = np.transpose(x)
        tempd0 = np.reshape(d0, (np.size(d0), 1))
        dw0 = learning_rate * tempx * tempd0

        # d1 = (y - op1) * op1 * (1 - op1)
        # dw1 = np.multiply(learning_rate, np.transpose(np.multiply(d1, op0)))
        # fnet = np.transpose(np.multiply(d(op0), d1))
        # d0 = np.sum(np.multiply(fnet, dw1), axis=0)[:-1]
        # temp = np.repeat(np.transpose(x).reshape(1, (np.size(x, axis=0))), len_hidden_layer, axis=0)
        # dw0 = np.multiply(learning_rate, np.multiply(np.reshape(d0, (len_hidden_layer, 1)), temp))

        self.w0 = np.add(self.w0, dw0)
        self.w1 = np.add(self.w1, dw1)
        pass

    def test_iteration(self, x, out_size=3, len_hidden_layer=5):
        # add the bias
        x = np.append(x, 1).reshape((np.size(x, axis=0) + 1, 1))
        if self.w0 is None:
            self.w0 = np.random.randn(len_hidden_layer, np.size(x, axis=0))
        net0 = np.matmul(self.w0, x)
        op0 = np.append(o(net0), 1).reshape((len_hidden_layer + 1, 1))

        if self.w1 is None:
            self.w1 = np.random.randn(out_size, np.size(op0, axis=0))
        net1 = np.matmul(self.w1, op0)
        op1 = o(net1).reshape((out_size,))
        return np.argmax(op1)

    def train(self, f, l):
        """
        :type f : features Matrix
        :type l : labels Matrix
        """
        f_data = f.data
        l_data = l.data
        data = np.hstack((f_data, l_data))
        # self.train_iteration([0, 0], 1, 1, 2, 1)
        # self.train_iteration([0, 1], 0, 1, 2, 1)
        for j in range(10000):
            for i in range(np.size(l.data)):
                self.train_iteration(f_data[i], np.array(refactor(l_data[i])))
            # shuffle the data
            np.random.shuffle(data)
            f_data = np.delete(data, np.s_[-1:], axis=1)
            l_data = np.delete(data, np.s_[:f.cols], axis=1)
        pass


    def predict(self, f, l):  # test
        """
        :type f: [float, ...]
        :type l: [float]
        """
        del l[:]
        l += [self.test_iteration(f)]
