from __future__ import (absolute_import, division, print_function, unicode_literals)

from toolkit.matrix import Matrix
from toolkit.supervised_learner import SupervisedLearner

import numpy as np
import matplotlib.pyplot as plt


def o(net):
    return 1 / (1 + np.exp(-net))

o = np.vectorize(o)

def d(s):
    return s * (1 - s)

d = np.vectorize(d)


# def refactor(l0):
#     if l0[0] == 0.0:
#         return [1, 0, 0]
#     elif l0[0] == 1:
#         return [0, 1, 0]
#     else:
#         return [0, 0, 1]
#     pass
#
def refactor(l0):
    if l0[0] == 0.0:
        return [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    elif l0[0] == 1:
        return [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    elif l0[0] == 2:
        return [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    elif l0[0] == 3:
        return [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    elif l0[0] == 4:
        return [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    elif l0[0] == 5:
        return [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    elif l0[0] == 6:
        return [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    elif l0[0] == 7:
        return [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
    elif l0[0] == 8:
        return [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    elif l0[0] == 9:
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    elif l0[0] == 10:
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
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
        self.dw0 = None
        self.dw1 = None
        self.mse_vs = []
        pass

    def train_iteration(self, x, y, out_size=3, len_hidden_layer=8, learning_rate=.01, momentum=0.0):
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

        if self.dw0 is None:
            self.dw0 = np.zeros((len_hidden_layer, np.size(x, axis=0)))
        if self.dw1 is None:
            self.dw1 = np.zeros((out_size, np.size(op0, axis=0)))

        # calc determinants
        d1 = (y - op1) * op1 * (1 - op1)
        tempO0 = np.repeat(op0.reshape((1, len_hidden_layer+1)), out_size, axis=0)
        tempd1 = np.repeat(d1, len_hidden_layer + 1, axis=0).reshape((out_size, len_hidden_layer+1))
        dw1 = np.multiply(np.multiply(tempO0, tempd1), learning_rate)
        dw1 += np.multiply(momentum, self.dw1)

        d0 = (np.sum(np.multiply(dw1, tempd1), axis=0) * np.transpose(op0) * (1 - np.transpose(op0)))[0][:-1]
        tempx = np.transpose(x)
        tempd0 = np.reshape(d0, (np.size(d0), 1))
        dw0 = learning_rate * tempx * tempd0
        dw0 += np.multiply(momentum, self.dw0)

        self.w0 = np.add(self.w0, dw0)
        self.w1 = np.add(self.w1, dw1)
        self.dw0 = dw0
        self.dw1 = dw1
        pass

    def test_iteration(self, x, y=None, out_size=3, len_hidden_layer=8):
        # add the bias
        x = np.append(x, 1).reshape((np.size(x, axis=0) + 1, 1))
        if self.w0 is None:
            self.w0 = np.random.rand(len_hidden_layer, np.size(x, axis=0))
        net0 = np.matmul(self.w0, x)
        op0 = np.append(o(net0), 1).reshape((len_hidden_layer + 1, 1))

        if self.w1 is None:
            self.w1 = np.random.rand(out_size, np.size(op0, axis=0))
        net1 = np.matmul(self.w1, op0)
        op1 = o(net1).reshape((out_size,))
        if y is not None:
            z = refactor(y)
            self.mse_vs += [1 / np.size(z) * np.sum(np.square(np.subtract(z, op1)))]
        return np.argmax(op1)

    def train(self, f, l):
        """
        :type f : features Matrix
        :type l : labels Matrix
        """
        data = np.hstack((f.data, l.data))
        np.random.shuffle(data)
        data, vs = np.vsplit(data, 2)
        f_vs = np.delete(vs, np.s_[-1:], axis=1)
        l_vs = np.delete(vs, np.s_[:f.cols], axis=1)

        f_data = np.delete(data, np.s_[-1:], axis=1)
        l_data = np.delete(data, np.s_[:f.cols], axis=1)
        # self.train_iteration([0, 0], 1, 1, 2, 1)
        # self.train_iteration([0, 1], 0, 1, 2, 1)
        bssf = 0.0
        k = 0
        acc_v = []
        mse_vs_v = []
        mse_tr_v = []
        mom = 0.7
        lr = .001
        for j in range(100000):
            for i in range(np.size(l_data)):
                # self.train_iteration(f_data[i], np.array(refactor(l_data[i])), momentum=0.0)
                self.train_iteration(f_data[i], np.array(refactor(l_data[i])), momentum=mom, learning_rate=lr,
                                     out_size=11, len_hidden_layer=80)
            # shuffle the data
            np.random.shuffle(data)
            f_data = np.delete(data, np.s_[-1:], axis=1)
            l_data = np.delete(data, np.s_[:f.cols], axis=1)
            acc = 0.0
            if np.mod(j, 100) == 0:
                for i in range(np.size(l_data)):
                    # self.test_iteration(f_data[i], l_data[i])
                    self.test_iteration(f_data[i], l_data[i], out_size=11, len_hidden_layer=80)
                mse = np.sum(self.mse_vs) / i
                mse_tr_v += [mse]
                self.mse_vs = []

                for i in range(np.size(l_vs)):
                    # if self.test_iteration(f_vs[i], l_vs[i]) == l_vs[i]:
                    if self.test_iteration(f_vs[i], l_vs[i], out_size=11, len_hidden_layer=80) == l_vs[i]:
                        acc += 1
                mse = np.sum(self.mse_vs) / i
                mse_vs_v += [mse]
                self.mse_vs = []

                acc /= np.size(l_vs)
                acc_v += [acc]
                print(acc)
                if acc > bssf:
                    bssf = acc
                    bssfw0 = self.w0
                    bssfw1 = self.w1
                    k = 0
                else:
                    k += 1
                    if k > 5 and mom == .7:
                        # lr = .01
                        mom = .03
                        k = 0
                        self.w0 = bssfw0
                        self.w1 = bssfw1
                        bssf = 0
                        print (j)
                    elif k > 10 and mom == .3:
                        # lr = .001
                        mom = 0.0
                        k = 0
                        self.w0 = bssfw0
                        self.w1 = bssfw1
                        bssf = 0
                        print (j)
                    elif k > 25:
                        print (j)
                        break
        self.w0 = bssfw0
        self.w1 = bssfw1

        plt.figure()
        plt.plot(acc_v)
        plt.xlabel("Epochs")
        plt.ylabel("Classification of accuracy (%)")
        plt.legend(["CLAS_ACC"])
        plt.show()

        plt.figure()
        plt.plot(mse_tr_v)
        plt.plot(mse_vs_v)
        plt.xlabel("Epochs")
        plt.ylabel("MSE")
        plt.legend(["MSE_TRS", "MSE_VS"])
        plt.show()
        pass


    def predict(self, f, l):  # test
        """
        :type f: [float, ...]
        :type l: [float]
        """
        del l[:]
        # l += [self.test_iteration(f)]
        l += [self.test_iteration(f, out_size=11, len_hidden_layer=80)]
