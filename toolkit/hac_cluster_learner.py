from __future__ import (absolute_import, division, print_function, unicode_literals)

from toolkit.supervised_learner import SupervisedLearner
from toolkit.matrix import Matrix
import numpy as np
import sys

def replace_inf(x, z):
    if np.isinf(x) or np.isneginf(x) or np.isnan(x):
        return z
    return x

replace_inf = np.vectorize(replace_inf)

def is_inf(x):
    if np.isinf(x) or np.isneginf(x) or np.isnan(x):
        return 0
    return 1

is_inf = np.vectorize(is_inf)

class HACClusterLearner(SupervisedLearner):
    """
    For nominal labels, this model simply returns the majority class. For
    continuous labels, it returns the mean value.
    If the learning model you're using doesn't do as well as this one,
    it's time to find a new learning model.
    """

    def __init__(self):
        self.k = 7
        self.complete_link = True
        self.normalization = False

        self.clusters = []
        self.features = []
        self.labels = []
        self.distance_matrix = []
        self.categorical = []
        self.max_features = []
        self.min_features = []
        pass

    def calc_distance_single(self, x, y):
        bssf = None
        for i in x:
            for j in y:
                distance = self.distance_matrix[i][j]
                if bssf is None or distance < bssf:
                    bssf = distance
        return bssf  # return distance between two given nodes

    def calc_distance_complete(self, x, y):
        bssf = None
        for i in x:
            for j in y:
                distance = self.distance_matrix[i][j]
                if bssf is None or distance > bssf:
                    bssf = distance
        return bssf  # return distance between two given nodes

    def find_shortest_distance(self):
        bssf = None
        cluster1 = None
        cluster2 = None
        for i in range(np.size(self.clusters)):
            for j in range(np.size(self.clusters)):
                if i == 25 and j == 34:
                    pass
                if j > i:
                    if self.complete_link:
                        distance = self.calc_distance_complete(self.clusters[i], self.clusters[j])
                    else:
                        distance = self.calc_distance_single(self.clusters[i], self.clusters[j])
                    if bssf is None or distance < bssf:
                        bssf = distance
                        cluster1 = i
                        cluster2 = j
                        pass
        return cluster1, cluster2  # return the two nodes that have the smallest distance


    def combine_clusters(self, x, y):
        new_clusters = []
        for i in range(np.size(self.clusters, axis=0)):
            if i != x and i != y:
                new_clusters.append(self.clusters[i])
            elif i == x:
                new_clusters.append(list(self.clusters[x]) + list(self.clusters[y]))
        self.clusters = np.array(list(new_clusters))
        return self.clusters

    def calc_distance(self, x, y):
        z = np.array(x) - np.array(y)
        for i in range(np.size(z)):
            if np.isinf(z[i]) or np.isneginf(z[i]) or np.isnan(z[i]):
                z[i] = self.max_features[i]
                # z[i] = 1
            elif self.categorical[i] and z[i] != 0:
                z[i] = 1

        # z[0] = 0  # todo: remove this
        return np.sum(np.square(z))

    def create_distance_matrix(self):
        self.distance_matrix = np.zeros((np.size(self.clusters), np.size(self.clusters)))
        for i in range(np.size(self.clusters)):
            self.distance_matrix[i][i] = 0
            for j in range(np.size(self.clusters)):
                if j > i:
                    z = self.calc_distance(self.features.data[i], self.features.data[j])
                    self.distance_matrix[i][j] = z
                    self.distance_matrix[j][i] = z
        return self.distance_matrix

    def calc_centroid(self, cluster):
        f_data = []
        if np.size(cluster) == 1:
            data = replace_inf(self.features.data, float('nan'))
            centroid = data[cluster[0]]
            centroid2 = np.copy(centroid)
            for j in range(np.size(self.categorical)):
                centroid = list(centroid)
                if self.categorical[j] and not np.isnan(centroid[j]):
                    centroid[j] = str(self.features.enum_to_str[j][centroid[j]])
            return centroid, centroid2

        centroid = np.zeros(self.features.cols)
        count = np.zeros(self.features.cols)
        data = replace_inf(self.features.data, float(0))
        for i in cluster:
            f_data.append(self.features.data[i])
            centroid += data[i]
            count += is_inf(self.features.data[i])
        centroid /= count
        centroid2 = np.copy(centroid)
        for j in range(np.size(self.categorical)):
            if self.categorical[j]:
                c = list(np.ndarray.astype(replace_inf(f_data, float(self.features.cols)), int))
                b = np.transpose(c)[j]
                a = np.bincount(b)
                if np.size(a) > 1:
                    a = a[:-1]
                else:
                    pass
                counts = np.argmax(a)
                centroid = list(centroid)
                centroid2 = list(centroid2)
                if np.sum(b) == 0:
                    centroid[j] = float('nan')
                    centroid2[j] = float('nan')
                else:
                    centroid[j] = str(self.features.enum_to_str[j][counts])
                    centroid2[j] = counts
                pass
        return centroid, centroid2

    def sse(self, cluster, centroid):
        sse = 0
        for c in cluster:
            error = self.calc_distance(centroid, self.features.data[c])
            sse += error
        return sse

    def calc_average_dissimilarity(self, i, cluster):
        diss = 0
        for x in cluster:
            diss += self.calc_distance(self.features.data[x], self.features.data[i])
        return diss / np.size(cluster, axis=0)

    def sil(self, clusters):
        total_score = 0
        total_count = 0
        for cluster in clusters:
            for i in range(np.size(cluster, axis=0)):
                a_i = self.calc_average_dissimilarity(i, cluster)
                b_i = None
                for cluster2 in clusters:
                    b_p = self.calc_average_dissimilarity(i, cluster2)
                    if (b_i is None or b_i > b_p) and b_i != a_i:
                        b_i = b_p

                total_score += (b_i - a_i) / np.maximum(b_i, a_i)
                total_count += 1

        return total_score/total_count

    def train(self, features, labels):
        """
        :type features: Matrix
        :type labels: Matrix
        """
        features.normalize()
        features.shuffle()
        self.features = features
        # self.features.data = features.data[1:100]
        self.labels = labels
        self.clusters = np.reshape(np.arange(np.size(labels.data)), (np.size(labels.data), 1))
        self.categorical = [bool(x) for x in features.str_to_enum]
        self.min_features = np.min(features.data, axis=0)
        self.features.data = replace_inf(self.features.data, float("-inf"))
        self.max_features = np.max(features.data, axis=0)


        self.distance_matrix = self.create_distance_matrix()
        while self.k != 1:
            while np.size(self.clusters, axis=0) > self.k:
                x, y = self.find_shortest_distance()
                self.combine_clusters(x, y)
            # for cluster in self.clusters:
            #     cluster.sort()
            centroids = []
            print('Number of Clusters : ' + str(self.k))
            total_sse = 0
            total_sil = self.sil(self.clusters)
            for cluster in self.clusters:
                print('Cluster: ' + str(cluster))
                centroid = self.calc_centroid(cluster)
                print('\tCentroid Values: ' + str(centroid[0]))
                print('\tSize: ' + str(np.size(cluster)))
                sse = self.sse(cluster, centroid[1])
                print('\tSSE: ' + str(sse))
                total_sse += sse
                centroids += [centroid]
            print('Total SSE: ' + str(total_sse))
            print('Total Silhouette: ' + str(total_sil))
            print()
            self.k -= 1
        sys.exit()

    def predict(self, features, labels):
        """
        :type features: [float]
        :type labels: [float]
        """
        del labels[:]
        labels += [1]
        # invalid!!!




