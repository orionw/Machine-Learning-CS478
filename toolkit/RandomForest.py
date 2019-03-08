from toolkit.supervised_learner import SupervisedLearner
from toolkit.matrix import Matrix
import numpy as np
import pandas as pd
from .DecisionTree import DecisionTree
from collections import Counter


class RandomForest(SupervisedLearner):

    def __init__(self, k_trees=99, bootstrap=True, percent_strap=.5):
        self.tree_list: list = []
        self.k_trees = k_trees
        if not bootstrap:
            self.percent_strap = 1
        else:
            self.percent_strap = percent_strap
        self.output_classes: int


    def train(self, features, labels):
        """
        :type features: Matrix
        :type labels: Matrix
        """
        # get the tree's ready
        for class_type in range(0, self.k_trees):
            self.tree_list.append(DecisionTree(pruning=True))

        # train on data
        for index, tree in enumerate(self.tree_list):
            boostrap_features, boostrap_labels = self.get_bootstrap_datasets(features, labels)
            tree.train(boostrap_features, boostrap_labels)

        print("Tree's formed")

    # def find_majority(self, votes):
    #     vote_count = Counter(votes)
    #     top_two = vote_count.most_common(2)
    #     if len(top_two) > 1 and top_two[0][1] == top_two[1][1]:
    #         # It is a tie - randomly decide
    #         if np.random.rand(1) > .5:
    #             return top_two[0][0]
    #         else:
    #             return top_two[1][0]
    #     return top_two[0][0]
    def measure_accuracy(self, features, labels, confusion=None, MSE=False):
        return super(RandomForest, self).measure_accuracy(features, labels, MSE=MSE)

    def find_majority(self, votes):
        # how many think it is theirs
        myMap = {}
        maximum = ('', 0)  # (occurring element, occurrences)
        for n in votes:
            value = n[0]
            if value in myMap:
                myMap[value] += 1
            else:
                myMap[value] = 1
            # Keep track of maximum on the go
            if myMap[value] > maximum[1]: maximum = (value, myMap[value])

        return maximum

    def predict(self, features, labels):
        """
        :type features: [float]
        :type labels: [float]
        """
        output_predictions = []
        for index, tree in enumerate(self.tree_list):
            output = tree.predict(features, labels)
            if type(output) == pd.Series or (type(output) == list and type(output[0]) == pd.Series):
                output = output[0].tolist()
            output_predictions.append(output)
        # get the majority vote from the list
        majority_vote = self.find_majority(output_predictions)[0]
        return_list = []
        return_list.append(majority_vote)
        return return_list

    def get_bootstrap_datasets(self, features, labels):
        # make copies to not affect original
        new_features = Matrix(matrix=features, row_start=0, col_start=0, row_count=features.rows,
                              col_count=features.cols)
        new_labels = Matrix(matrix=labels, row_start=0, col_start=0, row_count=labels.rows,
                            col_count=labels.cols)
        new_features.shuffle(buddy=new_labels)

        # take random samples of the dataset for this particular tree
        train_size = int(self.percent_strap * new_features.rows)
        bootstrap_features = Matrix(new_features, 0, 0, train_size, new_features.cols - 1)
        bootstrap_labels = Matrix(new_labels, 0, new_labels.cols - 1, train_size, 1)
        return bootstrap_features, bootstrap_labels


