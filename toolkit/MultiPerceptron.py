from toolkit.supervised_learner import SupervisedLearner
from toolkit.matrix import Matrix
import numpy as np
from .perceptron import Perceptron
from collections import Counter


class MultiPerceptron(SupervisedLearner):

    def __init__(self):
        self.perceptron_list: list = []
        self.output_classes: int


    def train(self, features, labels):
        """
        :type features: Matrix
        :type labels: Matrix
        """
        # get the perceptron's ready
        self.output_classes = len(set(labels.col(labels.cols - 1)))
        for class_type in range(0, self.output_classes):
            self.perceptron_list.append(Perceptron())

        # get data with right labels
        label_list: list = self.get_label_list(labels)

        # train on data
        for class_type in range(0, self.output_classes):
            self.perceptron_list[class_type].train(features, label_list[class_type])


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


    def find_majority(self, votes):
        # how many think it is theirs
        count = votes.count(1)
        if count > 1:
            values = np.array(votes)
            choice = 1 if np.random.rand(1) > .5 else 0
            index = [index for index, value in enumerate(values) if value == 1][choice]
            return index
        elif count == 0:
            return np.random.randint(0, self.output_classes + 1)
        # only one value, return it
        return votes.index(1)

    def predict(self, features, labels):
        """
        :type features: [float]
        :type labels: [float]
        """
        output_predictions = []
        for class_type in range(0, self.output_classes):
            output_predictions.append(self.perceptron_list[class_type].predict(features, labels)[0])
        # get the majority vote from the list
        majority_vote = self.find_majority(output_predictions)
        return_list = []
        return_list.append(majority_vote)
        return return_list

    def get_label_list(self, labels):
        label_list = []
        for count in range(self.output_classes):
            label_list.append(Matrix(labels, 0, 0, labels.rows, labels.cols))
        for row_num in range(labels.rows):
            output_class = int(labels.row(row_num)[0])
            for index in range(0, len(label_list)):
                label_list[index].set(row_num, 0, 1 if output_class == index else 0)
        return label_list


