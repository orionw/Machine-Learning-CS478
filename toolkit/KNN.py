from copy import deepcopy

from toolkit.supervised_learner import SupervisedLearner
from toolkit.matrix import Matrix
import numpy as np
import matplotlib as plt
import operator
from scipy.io.arff import loadarff
import pandas as pd
from collections import deque
import heapq
from scipy.spatial import distance
from .perceptron import Perceptron

class KNN(SupervisedLearner):
    """
    A Decision Tree with the ID3 Algorithm
    """

    labels = []

    def __init__(self, k: int = 9, distance_metric="minkowsky", weight_distance=False, use_perceptron=True):
        print("Initializing with a K of {}".format(k))
        self.weight_distance = weight_distance
        self.validation_set = False
        self.k = k
        self.regression: bool
        self.distance_metric = distance_metric
        self.use_weighted_columns = use_perceptron


    def train(self, features, labels):
        """
        :type features: Matrix
        :type labels: Matrix
        """
        # LAZY LEARNER
        if labels.value_count(0) == 0:
            self.regression = True
        else:
            self.regression = False

        if self.use_weighted_columns:
            new_feats = Matrix(features, 0, 0, features.rows, features.cols)
            self.perceptron = Perceptron()
            self.perceptron.train(new_feats, labels)
            self.weights = self.perceptron.weights
        self.feature_types = self.get_semantic_types(features)
        self.training_features = features
        self.training_labels = labels

    def predict(self, features, labels):
        """
        :type features: [float]
        :type labels: [float]
        """
        k_points = self.get_k_closest_points(features)
        k_labels = []
        for value in k_points.keys():
            k_labels.append(self.training_labels.row(value)[0])
        k_labels = np.array(k_labels)
        output = self.get_predicted_value(k_points, k_labels)
        return [output]

    def measure_accuracy(self, features, labels, confusion=None, MSE=False):
        return super(KNN, self).measure_accuracy(features, labels, MSE=MSE)

    """
    Tells whether each column is continuous or discrete
    """
    def get_semantic_types(self, features):
        features_type = []
        for col_num in range(features.cols):
            if features.value_count(col_num) == 0:
                features_type.append("continuous")
            else:
                features_type.append("discrete")
        return features_type

    def get_k_closest_points(self, point_row):
        distance_points = {}
        for index, row in enumerate(self.training_features.data):
            distance = self.get_distance(row, point_row)
            distance_points[index] = distance

        # we need smallest distance
        k_closest = dict(heapq.nsmallest(self.k,
                                         ((value, key) for key, value in distance_points.items())))
        closest_points = {v: k for k, v in k_closest.items()}

        return closest_points

    def get_distance(self, row, point_row):
        # if "discrete" not in self.feature_types:
        #     # use numpy for speed if only REAL values
        #     if self.distance_metric == "euclidean":
        #         row = np.array(row)
        #         point_row = np.array(point_row)
        #         distance = np.linalg.norm(abs(row - point_row))
        #         return distance
        #
        #     elif self.distance_metric == "manhattan":
        #         row = np.array(row)
        #         point_row = np.array(point_row)
        #         distance = (np.abs(row - point_row).sum())
        #         return distance
        #
        #     else:
        #         print("No valid distance defined")
        #         raise Exception
        # else:
        # there are nominal variables, be more careful
        distance = self.get_distance_types(row, point_row, self.distance_metric)
        return distance



    def get_distance_types(self, row, point_row, dist_type="euclidean"):
        sum_distance = 0
        for index, value in enumerate(row):
            if self.use_weighted_columns:
                weight = abs(self.weights[index])
            else:
                weight = 1
            if value == float("inf") or point_row[index] == float("inf"):
                # with don't know values, default to max distance
                sum_distance += (1 * weight)
            elif self.feature_types[index] == "continuous":
                sum_distance += (abs(value - point_row[index])) * weight
            else:
                # if not self.regression:
                #     sum_distance = self.get_vdm(index, value, point_row[index])
                # else:
                # type is nominal, if they are the same add zero, otherwise add 1
                diff = (1 - int(value == point_row[index]))  * weight
                sum_distance += diff


        if dist_type == "euclidean":
            # take the sqrt for Euclidean distance
            dist = np.sqrt(sum_distance)
        elif dist_type == "manhattan":
            dist = sum_distance
        elif dist_type == "minkowsky":
            dist = np.power(sum_distance, 3)
        else:
            raise Exception
        return dist

    def get_inverse_dist(self, distance_values):
        distance_values = np.array(list(distance_values))
        return sum(1 / np.square(distance_values))

    def get_predicted_value(self, k_points, k_labels):
        if self.regression:
            if self.weight_distance:
                inverse_distance_sum = self.get_inverse_dist(k_points.values())
                weighted = k_labels / np.square(list(k_points.values()))
                pred = sum(weighted) / inverse_distance_sum
            else:
                pred = sum(list(k_points.values()))/self.k

        else:
            k_labels = self.get_labels_from_k_points(k_points)
            if self.weight_distance:
                inverse_distance = self.get_inverse_dist(k_points.values())
                pred = self.get_highest_label(inverse_distance, k_labels)

            else:
                pred = self.find_majority(k_labels)[0]

        return pred

    def find_majority(self, votes):
        # how many think it is theirs
        myMap = {}
        maximum = ('', 0)  # (occurring element, occurrences)
        for n in votes:
            if n in myMap:
                myMap[n] += 1
            else:
                myMap[n] = 1
            # Keep track of maximum on the go
            if myMap[n] > maximum[1]: maximum = (n, myMap[n])

        return maximum

    def get_labels_from_k_points(self, k_points):
        labels = []
        for key, value in k_points.items():
            labels.append(int(self.training_labels.row(key)[0]))
        return labels

    def get_highest_label(self, inverse_distance, k_labels):
        sum_distance = sum(inverse_distance)
        class_output = np.zeros(self.training_labels.value_count(0))
        for index, label in enumerate(k_labels):
            class_output[label] += inverse_distance[index]

        class_output /= sum_distance
        arg_max_class = np.argmax(class_output)
        return arg_max_class

    def get_vdm(self, index, value_row, value_pred):
        if value_pred == value_row:
            return 0

        # values are not the same:
        output_classes = np.zeros(2, self.training_labels.value_count(0))
        for index_row, row in enumerate(self.training_features.data):
            if row[index] not in [value_row, value_pred]:
                continue
            if row[index] == value_row:
                class_is = 0
            else:
                class_is = 1
            output_classes[class_is][self.training_labels.data[index][0]] += 1

        for row in output_classes:
            row /= len(row)

        vda = sum(np.square(output_classes[0] - output_classes[1]))
        return vda










