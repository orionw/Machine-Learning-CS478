from toolkit.supervised_learner import SupervisedLearner
from toolkit.matrix import Matrix
import numpy as np
import matplotlib as plt
import operator
from scipy.io.arff import loadarff

# raw_data = loadarff('Training Dataset.arff')
# df_data = pd.DataFrame(raw_data[0])


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
        fitted_tree = self._fit_tree(train_features, train_labels)


    def predict(self, features, labels):
        """
        :type features: [float]
        :type labels: [float]
        """
        del labels[:]
        labels += self.labels
        return labels

    def measure_accuracy(self, features, labels, confusion=None, MSE=None):
        return super(DecisionTree, self).measure_accuracy(features, labels, MSE=MSE)

    def get_training_sets(self, features, labels):
        features_bias = Matrix(features, 0, 0, features.rows, features.cols)
        features_bias = self.add_bias_to_features(features_bias)

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

        if not self.validation_set:
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

    def _fit_tree(self, train_features, train_labels):
        dataset_info = self._get_info(train_labels, "all")
        column_gain = []
        for col in range(train_features.cols):
            column_gain.append(dataset_info - self._get_info(train_features.col(col), train_labels.data))
            
        most_gain_index = column_gain.index(max(column_gain))

    def _get_info(self, column, labels):
        if type(labels) != Matrix and labels == "all":
            # this is the main dataset calculation
            output_classes = np.zeros(column.value_count(0))
            for value in column.data:
                # remove the list part
                value_real = value[0]
                output_classes[int(value_real)] += 1
            output_classes /= len(column.data)
            info = self._calculate_entropy(output_classes)
        else:
            labels = [item for sublist in labels for item in sublist]
            classes_prop = np.zeros((len(np.unique(labels)), len(np.unique(column))))
            total_proportions = np.zeros(len(np.unique(column)))
            for index, value in enumerate(column):
                label_ind = labels[index]
                classes_prop[int(label_ind)][int(value)] += 1
                total_proportions[int(value)] += 1
            # transpose and divide by the number in column class
            classes_prop / total_proportions.T
            # divide column counts by column overall to get proportion
            total_proportions /= len(column)
            info = self._calculate_entropy(classes_prop, total_proportions)
        return

    def _calculate_entropy(self, output_classes, total_proportions=None):
        info = 0
        if total_proportions is None:
            for proportion in output_classes:
                info -= proportion * np.log2(proportion)
        else:
            for index, proportion in enumerate(output_classes):
                info -= proportion * np.log2(proportion) * total_proportions[index]

        return info
            
        



