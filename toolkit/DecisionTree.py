from toolkit.supervised_learner import SupervisedLearner
from toolkit.matrix import Matrix
import numpy as np
import matplotlib as plt
import operator
from scipy.io.arff import loadarff
import pandas as pd




class DecisionTree(SupervisedLearner):
    """
    A Decision Tree with the ID3 Algorithm
    """

    labels = []

    def __init__(self):
        self.validation_set = False

    def train(self, features, labels):
        """
        :type features: Matrix
        :type labels: Matrix
        """
        test_features, test_labels, train_features, train_labels = self.get_training_sets(features, labels)
        self.fitted_tree = Tree(train_features, train_labels)
        print("Tree Fitted")


    def predict(self, features, labels):
        """
        :type features: [float]
        :type labels: [float]
        """
        return [self.fitted_tree.execute(features)]

    def measure_accuracy(self, features, labels, confusion=None, MSE=False):
        return super(DecisionTree, self).measure_accuracy(features, labels, MSE=MSE)

    def get_training_sets(self, features, labels):
        features_bias = Matrix(features, 0, 0, features.rows, features.cols)
        # features_bias = self.add_bias_to_features(features_bias)

        #### Prepare Validation Set ####
        if self.validation_set:
            features_bias.shuffle(buddy=labels)

            test_size = int(.1 * features_bias.rows)
            train_features = Matrix(features_bias, 0, 0, features_bias.rows - test_size, features_bias.cols)
            train_features = self.add_bias_to_features(train_features)
            train_labels = Matrix(labels, 0, 0, features_bias.rows - test_size, labels.cols)

            test_features = Matrix(features_bias, test_size, 0, test_size, features_bias.cols)
            test_features = self.add_bias_to_features(test_features)
            test_labels = Matrix(labels, test_size, 0, test_size, labels.cols)

            train_features = train_features.return_pandas_df()
            train_labels = train_labels.return_pandas_df()
            test_features = test_features.return_pandas_df()
            test_labels = test_labels.return_pandas_df()

        if not self.validation_set:
            features_bias = features_bias.return_pandas_df()
            labels = labels.return_pandas_df()

            train_features = features_bias
            test_features = features_bias
            train_labels = labels
            test_labels = labels

        return test_features, test_labels, train_features, train_labels

    @staticmethod
    def add_bias_to_features(features, numpy_array=False):
        if not numpy_array:
            for row_num in range(features.rows):
                row = features.row(row_num)
                if len(row) < features.cols + 1:
                    # add bias - always one
                    row.append(1)
            return features
        else:
            for row in features:
                row = numpy_array.append(row, [1])
            return features



class Node:

    def __init__(self, col_index, features, labels, depth):
        if col_index is None:
            self.start_node = True
        else:
            self.start_node = False

        self.rule_column_index = col_index
        if int(labels.nunique()) == 1:
            self.index_value_map = int(labels.iloc[:, 0].unique()[0])
        else:
            best_index_gain = self.get_info_gain(features, labels)
            self.index_value_map = {}
            if depth != 0:
                for index, unique_val in enumerate(features.iloc[:, best_index_gain].unique()):
                    cols_to_index = features.iloc[:, best_index_gain] == unique_val
                    self.index_value_map[index] = Node(best_index_gain,
                                                       features[cols_to_index],
                                                       labels[cols_to_index],
                                                       depth - 1)

        return

    def execute(self, row):
        if not self.is_leaf_node():
            if self.start_node:
                return self.index_value_map[0].execute(row)
            else:
                value = int(row[self.rule_column_index])
                if type(self.index_value_map) == dict:
                    return self.index_value_map[value].execute(row)
                else:
                    return self.index_value_map
        else:
            return self.index_value_map

    def get_info_gain(self, train_features, train_labels):
        dataset_info = self._get_info(np.array(train_labels), "all")
        column_gain = []
        for col in train_features:
            if len(np.unique(train_features[col])) == 1:
                column_gain.append(float("-inf"))
            else:
                column_gain.append(dataset_info - self._get_info(np.array(train_features[col]),
                                                                 np.array(train_labels)))

        most_gain_index = column_gain.index(max(column_gain))
        return most_gain_index

    def _get_info(self, column, labels):
        if type(labels) != np.ndarray and labels == "all":
            # this is the main dataset calculation
            output_classes = np.zeros(len(np.unique(column)))
            for value in column:
                # remove the list part
                output_classes[int(value)] += 1
            output_classes /= len(column)
            info = self._calculate_entropy(output_classes)
        else:
            labels = [item for sublist in labels for item in sublist]
            classes_prop = np.zeros((len(np.unique(labels)), len(np.unique(column))))
            total_proportions = np.zeros(len(np.unique(column)))
            mapper = np.unique(column)
            for index, value in enumerate(column):
                label_ind = labels[index]
                classes_prop[int(label_ind)][np.where(mapper==int(value))] += 1
                total_proportions[np.where(mapper==int(value))] += 1
            # transpose and divide by the number in column class
            classes_prop /= total_proportions
            # divide column counts by column overall to get proportion
            total_proportions /= len(column)
            info = self._calculate_entropy(classes_prop, total_proportions)
        return info

    def _calculate_entropy(self, output_classes, total_proportions=None):
        info = 0
        if total_proportions is None:
            for proportion in output_classes:
                info -= proportion * np.log2(proportion)
        else:
            for index, proportion in enumerate(output_classes.T):
                next_info = np.dot(proportion, np.log2(proportion)) * total_proportions[index]
                if np.isnan(next_info):
                    next_info = 0
                info -= next_info

        return info

    def is_leaf_node(self):
        return type(self.index_value_map) == int


class Tree:

    def __init__(self, features, labels):
        self.start_node = Node(None, features, labels, depth=100)

    def execute(self, row):
        return self.start_node.execute(row)

