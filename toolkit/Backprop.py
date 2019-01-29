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
        # bias weights are moved to the end
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
                self._back_propagate(output, labels.row(row_num), row)
                print("Done backprop.  Weights are now: \n {}".format(self.weights))
                #accuracy_for_epoch: float = self._measure_accuracy(self.predict(features_bias, labels), labels)
                #self.accuracy_hash[str(self.epoch_count)] = accuracy_for_epoch
                self.epoch_count += 1
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
        return [0 * labels.rows]

    def _feed_forward(self, row):
        # TODO: update output matrix to hold all outputted outputs
        self.output_matrix = np.zeros(tuple([self.hidden_layers + 1, self.nodes_per_layer]))
        # send input as first input layer
        input_values = row
        next_input_values: list
        for index_matrix, weight_matrix in enumerate(self.weights):
            next_input_values = []
            for index_node, row in enumerate(weight_matrix):
                #self._build_output(input_values, row, index_matrix, index_node)
                net = np.dot(input_values, row)
                output = self.output_sigmoid(net)
                self.output_matrix[index_matrix][index_node] = output
                next_input_values.append(float(output))
            # add bias node if it's not the output layer
            if index_matrix != self.output_layers:
                next_input_values.append(1)
                # send current rows output to the next row
                input_values = next_input_values

        return next_input_values

    @staticmethod
    def _delta_output(target, output):
        return (target - output) * output * (1 - output)

    def _delta_hidden(self, output, matrix_index, weight_matrix):
        # output (1- output) * sum of weights to next layer * delta of that next layer
        # get the layer before delta values
        deltas = []
        for index, row in enumerate(weight_matrix[:, :-1].T):
            deltas.append(np.dot(row, self.delta_values_stored[matrix_index - 1]))
        deltas = np.array(deltas)
        return  output * (1 - output) * deltas

    def _back_propagate(self, output, target, input):
        # go through in reverse order to propagate error
        changes_in_weights = np.zeros_like(self.weights)
        # use np for array operations
        output = np.array(output)
        target = np.array(target)
        matrix_num = len(self.weights)
        self.delta_values_stored = np.zeros(tuple([self.hidden_layers + 1, self.nodes_per_layer]))
        # start from the end matrix -> output matrix
        for index_matrix, weight_matrix in enumerate(reversed(self.weights)):
            next_input_values = []

            # is output layer
            error = np.zeros([3, 2])
            if index_matrix == 0:
                delta = self._delta_output(target, output)
                # output layer goes in first
                self.delta_values_stored[index_matrix] = delta
                print("Delta for Output Nodes are {}".format(delta))
                for node_num in range(len(weight_matrix)):
                        # note: target and output are vectors
                        output_val = self.output_matrix[len(self.output_matrix) - 2 - index_matrix][node_num]
                        change_in_weights = self.learning_rate * delta * output_val

                        error[node_num] = change_in_weights
                # now do bias node by getting the bias weights to multiply in
                # bias nodes are learning_rate * delta  = change in weight
                error[len(weight_matrix[0]) - index_matrix - 1] = self.learning_rate * delta
                # update the weights
                changes_in_weights[len(self.weights) - index_matrix - 1] = error.T
                print("\nWeights for the output layer are: \n{}\n".
                      format(self.weights[matrix_num - index_matrix - 1] + error.T))

            # not output layer
            else:
                delta = self._delta_hidden(self.output_matrix[matrix_num - index_matrix - 1],
                                           index_matrix, self.weights[matrix_num - index_matrix])
                self.delta_values_stored[index_matrix] = delta
                print("Delta for hidden Nodes are {}".format(delta))

                for node_num in range(len(weight_matrix)):
                    # note: target and output are vectors
                    if matrix_num - index_matrix == 1:
                        # input layer, use input
                        output_val = input[node_num]
                    else:
                        output_val = self.output_matrix[len(self.output_matrix) - 2 - index_matrix][node_num]
                    change_in_weights = self.learning_rate * delta * output_val
                    error[node_num] = change_in_weights

                # now do bias node by getting the bias weights to multiply in
                # bias nodes are learning_rate * delta  = change in weight
                error[len(weight_matrix[0]) - 1] = self.learning_rate * delta
                # update the weights
                changes_in_weights[len(self.weights) - index_matrix - 1] = error.T
                print("\nWeights for layer {} are:\n {}\n".
                      format((matrix_num - index_matrix - 1),
                             self.weights[matrix_num - index_matrix - 1] + error.T))

        self.weights += changes_in_weights



