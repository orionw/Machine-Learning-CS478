from toolkit.supervised_learner import SupervisedLearner
from toolkit.matrix import Matrix
import numpy as np
import matplotlib as plt


class BackProp(SupervisedLearner):
    """
    A Multi-layered Neural Network using backpropagation
    """


    def __init__(self, nodes_per_layer=3, hidden_layers=1, output_nodes=1, batch_norm = False):
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
        self.batch_norm_enabled = batch_norm

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
        last_row_num = features_bias.rows - 1

        # start learning
        while self._is_still_learning(self):
            print(" #### On Epoch Number {} with weights \n{}\n".format(self.epoch_count, self.weights))
            for row_num in range(features_bias.rows):
                row = features_bias.row(row_num)
                print("Feed Forward with row: {}".format(row))
                output = self._feed_forward(row)
                print("backpropogating errors with output: {}".format(output))
                self._back_propagate(output, labels.row(row_num), row, self.batch_norm_enabled,
                                     (self.batch_norm_enabled and row_num == last_row_num))
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
        self.output_matrix = np.zeros(tuple([self.hidden_layers + 1, self.nodes_per_layer, self.hidden_layers + 1]))
        # send input as first input layer
        input_values = row
        next_input_values: list
        for index_matrix, weight_matrix in enumerate(self.weights):
            next_input_values = []
            output_row = []
            for index_node, row in enumerate(weight_matrix):
                #self._build_output(input_values, row, index_matrix, index_node)
                net = np.dot(input_values, row)
                output = self.output_sigmoid(net)
                output_row.append(output)
                if index_node == weight_matrix.shape[0] - 1:
                    output_row.append(1)
                next_input_values.append(float(output))
            # add bias node if it's not the output layer
            self.output_matrix[index_matrix] = np.tile(np.array(output_row), (2, 1))

            if index_matrix != self.output_layers:
                next_input_values.append(1)
                # send current rows output to the next row
                input_values = next_input_values

        return next_input_values

    @staticmethod
    def _delta_output(target, output):
        return (target - output) * output * (1 - output)

    def _delta_hidden(self, output, delta_previous, weight_matrix):
        # output (1- output) * sum of weights to next layer * delta of that next layer
        # get the layer before delta
        output = output[:, :-1]

        deltas = []
        for index, row in enumerate(weight_matrix[:, :-1].T):
            deltas.append(np.dot(row, delta_previous))
        deltas = np.array(deltas)
        return  output * (1 - output) * deltas

    def _back_propagate(self, output, target, input, batch_norm, end_of_batch=False):
        # use np for array operations
        output = np.array(output)
        target = np.array(target)
        matrix_num = len(self.weights)

        ################## Get Deltas #######################
        self.delta_storage = np.zeros_like(self.output_matrix)
        self.delta_storage[-1] = np.tile(self._delta_output(target, output), (3, 1)).T
        delta_prev = self.delta_storage[-1].T[0]
        for iteration in range(1, self.hidden_layers + 1):
            output_val = self.output_matrix[matrix_num - iteration - 1]
            delta_new = self._delta_hidden(output_val,
                                           delta_prev, self.weights[matrix_num - iteration])
            # reference not the last, but the one before it and then iterate downward
            self.delta_storage[self.delta_storage.shape[0] - 1 - iteration] = np.tile(delta_new[0], (3, 1)).T
            delta_prev = delta_new[0]

        # print("The error values (deltas) are \n{}".format(self.delta_storage))

        ################# Configure outputs ###################

        # shift the index one over to account for first output being recorded
        output_for_multi = np.copy(self.output_matrix)
        for value in range(0, self.output_matrix.shape[0] - 1):
            output_for_multi[value + 1] = self.output_matrix[value]
        output_for_multi[0] = np.tile(input, (2, 1))

        ##############W## Compute the weight changes ############
        weight_changes = self.delta_storage * output_for_multi * self.learning_rate

        if (batch_norm and end_of_batch) or not batch_norm:
            self.weights += weight_changes




