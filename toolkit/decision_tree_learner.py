from __future__ import (absolute_import, division, print_function, unicode_literals)

from toolkit.supervised_learner import SupervisedLearner
from toolkit.matrix import Matrix
import numpy as np

def count_var_with_name(x, name):
    if x == name:
        return 1
    else:
        return 0

count_var_with_name = np.vectorize(count_var_with_name)

class DecisionTreeLearner(SupervisedLearner):
    """
    For nominal labels, this model simply returns the majority class. For
    continuous labels, it returns the mean value.
    If the learning model you're using doesn't do as well as this one,
    it's time to find a new learning model.
    """

    labels = []

    def __init__(self):
        self.classes = []
        self.percent_for_training = 1
        self.info_s = 0
        self.head_node = self.Node()
        self.l_r = 0
        self.numNodes = 1

    class Node:

        def __init__(self):
            self.index_of_value = -1  # -1 = "leaf"
            self.num_child = 0
            self.child_array = []
            self.path = []
            self.predictive_label = -1
            self.size = 0
            self.vs_data_correct = 0
            self.vs_data_size = 0
            pass

        def add_child(self, node):
            self.num_child += 1
            self.child_array.append(node)
            pass

        def get_best_child(self):
            bssf = None
            bssf_i = -1
            for i in range(len(self.child_array)):
                if bssf is None or self.child_array[i].size > bssf:
                    bssf = self.child_array[i].size
                    bssf_i = i
            return bssf_i


    def create_new_node(self, head, bssf_i):
        self.numNodes += 1
        new_node = self.Node()
        new_node.predictive_label = head.predictive_label
        new_node.path = list(head.path)
        new_node.path.append(bssf_i)
        head.add_child(new_node)
        return head.child_array[-1]

    def idk(self, data, j, value):
        data_new = []
        for line in data:
            if line[j] == value:
                data_new += [line]
        return np.asarray(data_new)

    def calc_info_s(self, data):
        info = 0
        array_size = np.size(data, axis=0)
        for i in range(self.l_r):
            # noinspection PyTypeChecker
            num = np.sum(count_var_with_name(data, i), axis=0)[-1]
            p = num / array_size
            if p != 0:
                info -= p * np.log2(p)
        return info

    def calc_info_var(self, data, j):
        info = 0
        r = len(self.classes[j])
        array_size = np.size(data, axis=0)
        for i in range(r):
            # noinspection PyTypeChecker
            num = np.sum(count_var_with_name(data, i), axis=0)[j]
            p = num / array_size
            if p != 0:
                data_copy = self.idk(data, j, i)
                info += p * self.calc_info_s(data_copy)
        return info

    def most_common_label(self, data, head):
        if data.size == 0:
            return head.predictive_label
        new_data = np.delete(data, np.s_[:np.size(data, axis=1)-1], axis=1)
        new_data = np.ndarray.astype(np.reshape(new_data, (np.size(new_data))), int)
        counts = np.bincount(new_data)
        return np.argmax(counts)

    def expand_tree(self, data, head_node):
        bssf = None
        bssf_i = 0
        head_node.predictive_label = self.most_common_label(data, head_node)
        head_node.size = data.size
        if data.size is 0:
            return
        for i in range(np.size(data, axis=1) - 1):
            if i not in head_node.path:
                sol = self.calc_info_var(data, i)
                if bssf is None or sol < bssf:
                    bssf = sol
                    bssf_i = i
        if bssf is None:
            return
        head_node.index_of_value = bssf_i
        for i in range(len(self.classes[bssf_i])):
            new_node = self.create_new_node(head_node, bssf_i)
            self.expand_tree(self.idk(data, bssf_i, i), new_node)
        return


    def prune_tree(self, data, head, val):
        if np.size(head.child_array) != 0:
            the_bool = True
            val = head.predictive_label
            for i in range(np.size(head.child_array)):
                the_bool *= self.prune_tree(self.idk(data, head.index_of_value, i), head.child_array[i], val)
            if the_bool:
                head.child_array = []
            return the_bool
        else:
            if head.predictive_label != self.most_common_label(data, head):
                head.predictive_label = val
                self.numNodes -= 1
                return True
            return False
        pass

    def train(self, features, labels):
        """
        :type features: Matrix
        :type labels: Matrix
        """
        self.numNodes = 0
        self.classes = features.str_to_enum
        split_i = int(np.size(labels.data) * self.percent_for_training)
        data = np.hstack((features.data, labels.data))
        np.random.shuffle(data)
        tr, vs = data[:split_i, :], data[split_i:, :]

        self.l_r = len(labels.str_to_enum[0])
        self.info_s = self.calc_info_s(tr)
        self.expand_tree(tr, self.head_node)
        # self.prune_tree(vs, self.head_node, 0)
        print("Training Accuracy=" + str(self.measure_accuracy(features, labels)))
        print(self.numNodes)
        return

    def traverse_tree(self, f, head):
        if np.size(head.child_array) == 0:
            return head.predictive_label
        if np.isinf(f[head.index_of_value]):
            child = head.child_array[head.get_best_child()]
        else:
            child = head.child_array[int(f[head.index_of_value])]
        return self.traverse_tree(f, child)


    def predict(self, features, labels):
        """
        :type features: [float]
        :type labels: [float]
        """
        del labels[:]
        labels += [self.traverse_tree(features, self.head_node)]






