from copy import deepcopy

from toolkit.supervised_learner import SupervisedLearner
from toolkit.matrix import Matrix
import numpy as np
import matplotlib as plt
import operator
from scipy.io.arff import loadarff
import pandas as pd
from collections import deque


class DecisionTree(SupervisedLearner):
    """
    A Decision Tree with the ID3 Algorithm
    """

    labels = []

    def __init__(self, pruning=False):
        self.validation_set = pruning
        self.pruning = pruning

    def train(self, features, labels):
        """
        :type features: Matrix
        :type labels: Matrix
        """
        test_features, test_labels, train_features, train_labels = self.get_training_sets(features, labels)
        self.fitted_tree = Tree(train_features, train_labels)

        best_accuracy = self.measure_accuracy_pandas(train_features, train_labels)
        print("Best Train Set Accuracy BEFORE is {}".format(best_accuracy))
        #print(self.fitted_tree)
        if self.pruning:
            self.fitted_tree = self.prune_and_find_best_tree(self.fitted_tree, test_features, test_labels)
        print("Tree Fitted")
        best_accuracy = self.measure_accuracy_pandas(train_features, train_labels)
        print("Best Train Set Accuracy is {}".format(best_accuracy))
        best_accuracy = self.measure_accuracy_pandas(test_features, test_labels)
        print("Best Test Set Accuracy is {}".format(best_accuracy))
        print("There are {} nodes active out of {} with a depth of {}".format(self.fitted_tree.get_depth()[1],
                                                                              self.fitted_tree.num_nodes,
                                                                              self.fitted_tree.depth))


    def predict(self, features, labels):
        """
        :type features: [float]
        :type labels: [float]
        """

        return [self.fitted_tree.execute(pd.Series(features))]

    def measure_accuracy(self, features, labels, confusion=None, MSE=False):
        return super(DecisionTree, self).measure_accuracy(features, labels, MSE=MSE)

    def measure_accuracy_pandas(self, features, labels, confusion=None, MSE=False):
        return super(DecisionTree, self).measure_accuracy_pandas(features, labels, MSE=MSE)

    def get_training_sets(self, features, labels):
        features_bias = Matrix(features, 0, 0, features.rows, features.cols)
        # features_bias = self.add_bias_to_features(features_bias)

        #### Prepare Validation Set ####
        if self.validation_set:
            features_bias.shuffle(buddy=labels)

            test_size = int(.4 * features_bias.rows)
            train_features = Matrix(features_bias, 0, 0, features_bias.rows - test_size, features_bias.cols)
            train_labels = Matrix(labels, 0, 0, features_bias.rows - test_size, labels.cols)

            test_features = Matrix(features_bias, test_size, 0, test_size, features_bias.cols)
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

    def prune_and_find_best_tree(self, fitted_tree, test_features, test_labels):
        accuracy_previous = self.measure_accuracy_pandas(test_features, test_labels)
        accuracy_after_prune = float("inf")
        prune_starting_at = 1
        accuracy_list = []
        best_tree: DecisionTree = self.fitted_tree
        best_accuracy: float = accuracy_previous
        retry = True
        prev_active_nodes = self.fitted_tree.get_depth()[1]
        print("initial accuracy: {}".format(best_accuracy))
        while retry:
            retry = False
            print("Retrying")
            for index in reversed(range(int(len(self.fitted_tree.node_list)))):
                if self.fitted_tree.node_list[index].is_leaf_node() or self.fitted_tree.node_list[index].off:
                    continue
                old_tree = deepcopy(best_tree)
                self.fitted_tree, active_nodes = self._prune(old_tree, index)
                active_nodes = self.fitted_tree.get_depth()[1]
                accuracy_after_prune = self.measure_accuracy_pandas(test_features, test_labels)
                print("Accuracy for {} is {}".format(index, accuracy_after_prune))
                accuracy_list.append(accuracy_after_prune)
                if accuracy_after_prune >= best_accuracy + .001 and active_nodes < prev_active_nodes:
                    if best_tree is self.fitted_tree:
                        raise KeyError
                    best_tree = deepcopy(self.fitted_tree)
                    print(best_tree.get_depth())
                    retry = True
                    prev_active_nodes = best_tree.get_depth()[1]
                    print("Test Set gain is {}. Accuracy is now: {} with {} active nodes".format(accuracy_after_prune - best_accuracy,
                                                                                                     accuracy_after_prune,
                                                                                                     active_nodes))
                    best_accuracy = accuracy_after_prune
                    break

        print(best_tree.get_depth())
        self.fitted_tree = best_tree
        print(self.fitted_tree.get_depth())

        # print("List of pruning is {}".format(accuracy_list))

        return self.fitted_tree

    def _prune(self, fitted_tree, prune_starting_at):
        from copy import deepcopy
        new_tree = deepcopy(fitted_tree)
        inactive = new_tree.prune(prune_starting_at)
        return new_tree, inactive


class Node:

    def __init__(self, col_index, features, labels, depth):
        self.off = False
        self.depth = depth
        if col_index is None:
            self.start_node = True
        else:
            self.start_node = False
        if features.isnull().values.any():
            real_features = features
            if not features.isnull().all().all():
                features = features.fillna(features.mode().iloc[0])
                # if any columns are still NA they are all NA.  Fill with zero
                features = features.fillna(0)
            else:
                # all values are NA, replace with zero (shouldn't happen)
                features = features.fillna(0)
                print("Bad dataset. Replacing with zeros")
        else:
            real_features = features
        self.majority_class = labels.mode().iloc[:, 0]
        if type(self.majority_class) == pd.Series and len(self.majority_class) > 1:
            rand_value = np.random.randint(0, len(labels.mode().iloc[:, 0]), 1)[0]
            self.majority_class = self.majority_class[rand_value]
        #print("Applicable instances are {}".format(labels.count()))
        # self.rule_column_index = col_index
        # second conditions checks that all rows are the same as first row
        if int(labels.nunique()) == 1 or (features.iloc[0] == features).all().all():
            #print("Is a leaf node: {}".format(features.columns[col_index]))
            self.index_value_map = int(labels.iloc[:, 0].unique()[0])
            #print("The leaf is {}".format(self.index_value_map))
        else:
            best_index_gain = self.get_info_gain(features, labels)
            self.rule_column_index = best_index_gain
            self.rule_column_name = features.columns[best_index_gain]
            self.index_value_map = {}
            if depth != float("inf"):
                for index, unique_val in enumerate(sorted(features.iloc[:, best_index_gain].unique())):
                    cols_to_index = features.iloc[:, best_index_gain] == unique_val
                    self.index_value_map[index] = Node(best_index_gain,
                                                       real_features[cols_to_index],
                                                       labels[cols_to_index],
                                                       depth + 1)

        return

    def execute(self, row):
        if not self.off:
            if not self.is_leaf_node():
                value = int(row[self.rule_column_index])
                if type(self.index_value_map) == dict:
                    try:
                        return self.index_value_map[value].execute(row)
                    except KeyError:
                        return self.majority_class
                else:
                    return self.index_value_map
            else:
                return self.index_value_map
        else:
            # node is off, might not be an easy split.  Return majority class
            # print("###################################################")
            return self.majority_class

    def get_info_gain(self, train_features, train_labels):
        dataset_info = self._get_info(np.array(train_labels), "all")
        #print("Dataset info is {}".format(dataset_info))
        column_gain = []
        for index, col in enumerate(train_features):
            if len(np.unique(train_features[col])) == 1:
                column_gain.append(float("-inf"))
            else:
                column_gain.append(dataset_info - self._get_info(np.array(train_features[col]),
                                                                 np.array(train_labels)))
                #print("Column gain for {} is {}".format(train_features.columns[index], column_gain[index]))

        most_gain_index = column_gain.index(max(column_gain))
        #print("Splitting on {}".format(train_features.columns[most_gain_index]))
        return most_gain_index

    def _get_info(self, column, labels):
        if type(labels) != np.ndarray and labels == "all":
            # this is the main dataset calculation
            output_classes = np.zeros(len(np.unique(column)))
            mapper = np.unique(column)
            for value in column:
                # remove the list part
                output_classes[np.where(mapper==int(value))] += 1
            output_classes /= len(column)
            info = self._calculate_entropy(output_classes)
        else:
            labels = [item for sublist in labels for item in sublist]
            classes_prop = np.zeros((len(np.unique(labels)), len(np.unique(column))))
            total_proportions = np.zeros(len(np.unique(column)))
            mapper = np.unique(column)
            labels_mapper = np.unique(labels)
            for index, value in enumerate(column):
                label_ind = labels[index]
                label_index = np.where(labels_mapper==int(label_ind))[0][0]
                column_index = np.where(mapper==int(value))[0][0]
                classes_prop[label_index][column_index] += 1
                total_proportions[column_index] += 1
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
                # make sure there are no -inf in the log proportions due to a zero
                log_prop = np.log2(proportion)
                log_prop[np.isneginf(log_prop)] = 0
                next_info = np.dot(proportion, log_prop ) * total_proportions[index]
                if np.isnan(next_info):
                    next_info = 0
                info -= next_info

        return info

    def is_leaf_node(self):
        return type(self.index_value_map) == int

    def turn_off_prune(self):
        self.off = True
        if type(self.index_value_map) == dict:
            for index, child_node in self.index_value_map.items():
                child_node.turn_off_prune()

class Tree:

    def __init__(self, features, labels):
        self.num_nodes = 0
        self.depth = 0
        self.start_node = Node(None, features, labels, depth=0)
        self.node_list = self.gather_nodes(self.start_node)
        self.depth, self.active_nodes = self.get_depth()
        print("Tree Formed")

    def execute(self, row):
        if row.isnull().values.any():
            if row.isnull().all():
                row = row.fillna(0)
            else:
                row = row.fillna(row.mode().iloc[0])
        return self.start_node.execute(row)

    def prune(self, prune_starting_at):
        self.node_list[prune_starting_at].turn_off_prune()
        inactive = 0
        for node in self.node_list:
            if node.off:
                inactive += 1
        # print("Done pruning. {} inactive Nodes".format(inactive))
        return inactive

    def gather_nodes(self, start_node):
        node_list = []
        queue = deque()
        queue.append(start_node)
        node_list.append(start_node)
        while len(queue):
            current = queue.pop()
            if type(current.index_value_map) == dict:
                for index, node in current.index_value_map.items():
                    queue.append(node)
                    node_list.append(node)
            else:
                # is leaf node
                pass
        self.num_nodes = len(node_list)
        return node_list

    def get_depth(self):
        max_depth = 0
        on_nodes = 0
        for node in self.node_list:
            if not node.off:
                on_nodes += 1
                if node.depth > max_depth:
                    max_depth = node.depth
        return max_depth, on_nodes
