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

class KMeansClusterLearner(SupervisedLearner):
    """
    For nominal labels, this model simply returns the majority class. For
    continuous labels, it returns the mean value.
    If the learning model you're using doesn't do as well as this one,
    it's time to find a new learning model.
    """

    labels = []

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
        self.centroids = []
        pass

    def calc_distance(self, x, y):
        z = np.array(x) - np.array(y)
        for i in range(np.size(z)):
            if np.isinf(z[i]) or np.isneginf(z[i]) or np.isnan(z[i]):
                # z[i] = self.max_features[i]
                z[i] = 1
            elif self.categorical[i] and z[i] != 0:
                z[i] = 1

        # z[0] = 0  # todo: remove this
        return np.sum(np.square(z))

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
                c = list(np.ndarray.astype(replace_inf(list(f_data), float(self.features.cols)), int))
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

    def calculate_new_clusters(self):
        self.clusters = []
        for i in range(self.k):
            self.clusters += [[]]
            pass
        i = 0
        for x in self.features.data:
            bssf = None
            j = 0
            for y in self.centroids:
                dist = self.calc_distance(x, y)
                if bssf is None or dist < bssf:
                    bssf = dist
                    k = j
                j += 1
            self.clusters[k] += [i]
            # print(str(i) + '=' + str(k))
            i += 1
        return self.clusters

    def calc_new_centroids(self):
        centroids = []
        self.print_cent = []
        for x in self.clusters:
            centroids += [self.calc_centroid(x)[1]]
            self.print_cent += [self.calc_centroid(x)[0]]
        self.centroids = centroids
        return self.centroids

    def calc_new_sse(self):
        total = 0
        for i in range(np.size(self.clusters, axis=0)):
            total += self.sse(self.clusters[i], self.centroids[i])
        return total

    def train(self, features, labels):
        """
        :type features: Matrix
        :type labels: Matrix
        """
        # features.normalize()
        # features.shuffle()
        self.features = features
        self.categorical = [bool(x) for x in features.str_to_enum]
        self.min_features = np.min(features.data, axis=0)
        self.features.data = replace_inf(self.features.data, float("-inf"))
        self.max_features = np.max(self.features.data, axis=0)
        # if self.normalization:
        #     for i in range(np.size(self.features.data, axis=0)):
        #         for j in range(np.size(self.features.data, axis=1)):
        #             if not np.isinf(self.features.data[i][j]) and not np.isneginf(self.features.data[i][j]) and \
        #                     not np.isnan(self.features.data[i][j]) and not self.categorical[j]:
        #                 self.features.data[i][j] = (self.features.data[i][j] - self.min_features[j]) /\
        #                                            (self.max_features[j] - self.min_features[j])

        for i in range(self.k):
            self.centroids += [np.copy(self.features.data[i])]
        print('Iteration ' + str(0))
        for x in self.centroids:
            print('Centroid: ' + str(np.array(x)))
        self.clusters = self.calculate_new_clusters()
        b_sse = self.calc_new_sse()
        print('SSE: ' + str(b_sse) + '\n')
        sse = 0
        i = 0
        while b_sse > sse or i < 20:
            b_sse = sse
            print('Iteration ' + str(i + 1))
            self.centroids = self.calc_new_centroids()
            for x in self.centroids:
            # for x in self.print_cent:
                print('Centroid: ' + str(x))
            self.clusters = self.calculate_new_clusters()
            sse = self.calc_new_sse()
            print('SSE: ' + str(sse))
            print()
            i += 1
            total_sil = self.sil(self.clusters)
            print('Total Silhouette: ' + str(total_sil))
        sys.exit()


    def predict(self, features, labels):
        """
        :type features: [float]
        :type labels: [float]
        """
        del labels[:]
        labels += self.labels



