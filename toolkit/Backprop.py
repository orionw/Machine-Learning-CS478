from toolkit.supervised_learner import SupervisedLearner
from toolkit.matrix import Matrix
import numpy as np
import matplotlib as plt


class BackProp(SupervisedLearner):
    """
    A Multi-layered Neural Network using backpropagation
    """


    def __init__(self, nodes_per_layer=3, hidden_layers=1, output_nodes=1):
        # add bias and initial weights
        # self.weights = []
        # for layer in range(hidden_layers):
        #     self.weights.append(np.random.rand(nodes_per_layer, nodes_per_layer + 1))
        self.weights = [[[0.2, -0.1, 0.1], [0.3, -0.3, -0.2]],
                        [[-0.2, -0.3, 0.1], [-0.1, 0.3, 0.2]],
                        [[-0.1, 0.3, 0.2], [-0.2, -0.3, 0.1]]]
        self.weights = np.asarray(self.weights)
        ## Note on Weights ##
        # Weights are organized backwards = the top of the array is the leftmost(bottom) layer
        self.output_layer: np.ndarray = np.random.rand(1)
        self.nodes_per_layer = nodes_per_layer
        self.hidden_layers = hidden_layers
        self.accuracy_hash = {}
        self.epoch_count = 0
        self.learning_rate: float = .1
        self.momentum: float = .1
        self.labels = []
        self.output_classes = 0
        self.input_layer = []

    def add_bias_to_features(self, features, numpy_array=False):
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

    def train(self, features, labels):
        """
        :type features: Matrix
        :type labels: Matrix
        """
        # setup input layer to match specs
        self.input_layer = np.random.rand(features.cols, features.cols + 1)
        # setup output layer to match specs
        # self.output_classes = len(set(labels.col(labels.cols - 1)))
        # self.output_layer: np.ndarray = np.random.rand(labels.cols)
        self.output_layers = labels.cols
        # create a new features set with a bias for training
        features_bias = Matrix(features, 0, 0, features.rows, features.cols)
        features_bias = self.add_bias_to_features(features_bias)

        # start learning
        while self._is_still_learning(self):
            print(" #### On Epoch Number {} with weights {}".format(self.epoch_count, self.weights))
            for row_num in range(features_bias.rows):
                row = features_bias.row(row_num)
                print("Feed Forward with row: {}".format(row))
                output = self._feed_forward(row)
                print("backpropogating errors with output: {}".format(output))
                self._back_propagate(output, labels.row(row_num))
                print("Done backprop.  Weights are now: {}".format(self.weights))
                accuracy_for_epoch: float = self._measure_accuracy(self.predict(features_bias, labels), labels)
                self.accuracy_hash[str(self.epoch_count)] = accuracy_for_epoch
        return


    def _measure_accuracy(self, preds, labels):
        labels = np.asarray(labels.data)
        preds = np.asarray(preds)

        assert(len(preds) == len(labels))
        correct: int = 0
        for i in range(len(preds)):
            if preds[i] == labels[i]:
                correct += 1
        print("Accuracy is {}% with weights {}".format(correct / len(preds), self.weights))
        return correct / len(preds)

    @staticmethod
    def _is_still_learning(self):
        margin_increasing = .01
        if len(self.accuracy_hash) <= 5:
            return True
        # check that the accuracy hasn't improved for 5 iterations
        baseline: float = self.accuracy_hash[str(self.epoch_count - 5)]
        # check the end point also
        for iteration in range(self.epoch_count - 4, self.epoch_count + 1):
            if self.accuracy_hash[str(iteration)] - baseline > margin_increasing:
                return True
        else:
            return False

    @staticmethod
    def output_sigmoid(net):
        return 1 / (1 + np.exp(-net))

    def predict(self, features, labels):
        """
        :type features: [float]
        :type labels: [float]
        """
        del labels[:]
        labels += self.labels
        return labels

    def _feed_forward(self, row):
        # TODO: update output matrix to hold all outputted outputs
        self.output_matrix = np.zeros(tuple([self.nodes_per_layer, self.hidden_layers + 1]))
        # send input as first input layer
        input_values = row
        next_input_values: list
        for index_matrix, weight_matrix in enumerate(self.weights):
            next_input_values = []
            for index_node, row in enumerate(weight_matrix):
                net = np.dot(input_values, row)
                output = self.output_sigmoid(net)
                self.output_matrix[index_node][index_matrix] = output
                next_input_values.append(float(output))
            # add bias node if it's not the output layer
            if index_matrix != self.output_layers:
                next_input_values.append(1)
                # send current rows output to the next row
                input_values = next_input_values

        return next_input_values

    def _back_propagate(self, output, target):
        # go through in reverse order to propagate error
        error = []
        # use np for array operations
        output = np.array(output)
        target = np.array(target)

        for index_matrix, weight_matrix in enumerate(reversed(self.weights)):
            next_input_values = []
            for index_node, row in enumerate(weight_matrix):
                # is output layer
                if index_matrix == 0:
                    for node_num in range(self.output_layers):
                        # TODO: this needs to get it for one node
                        delta = (target - output) * (output * (1 - output))
                        error.append(self.learning_rate * delta * self.output_matrix[-1][node_num])
                print("This is the backprop error: {}".format(error))



