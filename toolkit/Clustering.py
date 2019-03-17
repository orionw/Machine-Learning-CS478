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
from scipy.stats import mode as mode

class Cluster(SupervisedLearner):
    """
    A Decision Tree with the ID3 Algorithm
    """

    labels = []

    def __init__(self, type="HAC", k=3, link_type="single"):
        if k < 1:
            print("Bad K value. Quitting")
            raise Exception
        self.cluster_type = type
        if type == "K-Means":
            print("Initializing with a K of {}".format(k))
        else:
            print("Initializing with a K of up to {}".format(k))
        self.k = k
        self.distance_matrix: np.ndarray
        self.link_type = link_type



    def train(self, features, labels):
        """
        :type features: Matrix
        :type labels: Matrix
        """
        self.feature_types = self.get_semantic_types(features)
        features = features.return_pandas_df()
        if self.cluster_type == "HAC":
            self.cluster_map = {}
            # create cluster of every point by themselves
            prev_cluster_list = []
            for i in range(features.shape[0]):
                prev_cluster_list.append([i])
            clusters = features.shape[0]
            self.distance_matrix = self.get_distance_matrix(features)
            while clusters > self.k:
                # while there are more clusters than we want - combined a cluster
                clusters, prev_cluster_list, features = self.cluster_again(prev_cluster_list, features)
                self.cluster_map[clusters] = prev_cluster_list

        elif self.cluster_type == "K-Means":
            new_means = True
            means = self.get_init_centroids(features)
            count = 0
            prev_sse = -1
            while new_means:
                print("\n On iteration {}".format(count))
                count += 1
                new_means, means, clustering, total_sse = self.reevaluate_centroid(means, features)
                if total_sse == prev_sse:
                    new_means = False
                else:
                    prev_sse = total_sse
                    new_means = True


            print("Done clustering")



    def predict(self, features, labels):
        """
        :type features: [float]
        :type labels: [float]
        """
        pass

    def measure_accuracy(self, features, labels, confusion=None, MSE=False):
        print("What am I measuring accuracy on???")

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

    def get_distance_matrix(self, features):
        distance_matrix = np.full(tuple((features.shape[0], features.shape[0])), fill_value=float("inf"))
        for first_index, row in enumerate(features.iterrows()):
            for second_index, row_two in enumerate(features.iterrows()):
                if first_index > second_index:
                    # it's symmetric, skip
                    continue
                elif first_index == second_index:
                    distance_matrix[first_index][second_index] = float("inf")
                else:
                    distance = self.get_distance(row, row_two)
                    distance_matrix[first_index][second_index] = distance

        return distance_matrix

    def get_distance(self, row, point_row, dist_type="euclidean"):
        distance = self.get_distance_types(row, point_row, dist_type)
        return distance

    def get_distance_types(self, row, point_row, dist_type="euclidean"):
        sum_distance = 0
        # make them into series
        row = row[1]
        point_row = point_row[1]
        if point_row.equals(row):
            # they're the same row
            if self.cluster_type == "K-Means":
                return 0
            else:
                # remove this option in HAC
                return float("inf")

        for index, value in enumerate(row):
            if value == float("nan") or point_row[index] == float("nan") or\
                    np.isnan(value) or np.isnan(point_row[index]):
                # with don't know values, default to max distance
                sum_distance += 1
            elif self.feature_types[index] == "continuous":
                sum_distance += np.square(abs(value - point_row[index]))
            else:
                diff = (1 - int(value == point_row[index]))
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

    def cluster_again(self, prev_cluster_list, features: pd.DataFrame):
        # find the next cluster to merge
        # if they're the same, try again
        merged_cluster = False
        while not merged_cluster:
            # index of min element
            ind = np.unravel_index(np.argmin(self.distance_matrix, axis=None), self.distance_matrix.shape)
            value = self.distance_matrix[ind]
            # mark as seen
            self.distance_matrix[ind] = float("inf")
            first_to_merge = None
            second_to_merge = None
            for index_cluster, cluster in enumerate(prev_cluster_list):
                if ind[0] in cluster:
                    first_to_merge = index_cluster
                if ind[1] in cluster:
                    second_to_merge = index_cluster
                if first_to_merge is not None and second_to_merge is not None:
                    break

            if first_to_merge != second_to_merge:
                merged_cluster = True

        if self.link_type == "single":

            for row in range(self.distance_matrix.shape[0]):
                self.distance_matrix[row][first_to_merge] = min(self.distance_matrix[row][first_to_merge],
                                                                self.distance_matrix[row][second_to_merge])
                self.distance_matrix[row][second_to_merge] = float("inf")

        elif self.link_type == "complete":
            for row in range(self.distance_matrix.shape[1]):
                self.distance_matrix[first_to_merge][row] = max(self.distance_matrix[first_to_merge][row],
                                                                self.distance_matrix[second_to_merge][row])
                self.distance_matrix[second_to_merge][row] = float("inf")
                # self.distance_matrix[row][first_to_merge] = min(self.distance_matrix[row][first_to_merge],
                #                                                 self.distance_matrix[row][second_to_merge])
                # self.distance_matrix[row][second_to_merge] = float("inf")


        # Start the merger
        print('#############################################3')
        print("Merged cluster {} with {}, value of {}: \n containing {} and {}".format(first_to_merge, second_to_merge,
                                                                                       value,
                                                                                       prev_cluster_list[first_to_merge],
                                                                                       prev_cluster_list[second_to_merge]))
        # combine two lists and remove the other cluster
        prev_cluster_list[first_to_merge] += prev_cluster_list[second_to_merge]
        del prev_cluster_list[second_to_merge]
        for index, cluster in enumerate(prev_cluster_list):
            print("Cluster {}: {}".format(index, cluster))

        return len(prev_cluster_list), prev_cluster_list, features

    def get_init_centroids(self, features):
        means = []
        for index, row in enumerate(features.iterrows()):
            if index == self.k:
                break
            means.append(row)
        return means

    def reevaluate_centroid(self, means, features):
        clustering = [[] for x in range(len(means))]
        for index, row in enumerate(features.iterrows()):
            distances = []
            for centroid in means:
                dist = self.get_distance(row, centroid)
                distances.append(dist)

            closest_mean = distances.index(min(distances))
            clustering[closest_mean].append(index)

        new_centroids = self.find_centroids(clustering, features)
        means_changed = self.are_centroids_equal(means, new_centroids)
        cluster_sse, total_sse = self.get_sse(clustering, features)

        return means_changed, new_centroids, clustering, total_sse

    def find_centroids(self, clustering, features):
        new_mean_list = []
        for index, cluster in enumerate(clustering):
            new_centroid = features.iloc[cluster, :].apply(lambda x: x.mean())
            new_mean_list.append(tuple((index, new_centroid)))


        return new_mean_list

    def are_centroids_equal(self, means, new_centroids):
        for init_mean in means:
            for new_mean in new_centroids:
                if not init_mean[1].equals(new_mean[1]):
                    return True
        else:
            return False

    def get_sse(self, clusters, features):
        cluster_sse = []
        total_sse = 0
        for index_cluster, cluster in enumerate(clusters):
            centroid = self.get_mean_pandas(features.iloc[cluster, :])
            print("Centroid {}: {}".format(index_cluster, centroid))
            sse = 0
            for row_num in cluster:
                row = features.iloc[row_num, :]
                dist = self.get_distance(tuple((0, row)), tuple((1, centroid)), dist_type="manhattan")
                sse += dist
            cluster_sse.append(sse)
            total_sse += sse

        for index, cluster_ind_sse in enumerate(cluster_sse):
            print("Cluster {}: {} => SSE: {}".format(index, clusters[index], cluster_ind_sse))
        print("Total SSE is {}".format(total_sse))
        return cluster_sse, total_sse

    def get_mean_pandas(self, df):
        mean_df = pd.DataFrame(columns=df.columns)
        new_df = {}
        for index, column in enumerate(df.columns):
            if self.feature_types[index] == "discrete":
                mean = mode(df[column], nan_policy="propagate", axis=None)
                counts = mean[1]
                mean = mean[0]
                if type(mean) == np.ndarray:
                    mean = mean[0]
                # if counts <= df[column].isna().sum():
                #     mean = np.nan
            else:
                mean = df[column].mean()

            new_df[column] = mean

        mean_df = mean_df.append(new_df, ignore_index=True)
        return mean_df.iloc[0, :]















