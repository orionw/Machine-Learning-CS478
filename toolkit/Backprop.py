from toolkit.supervised_learner import SupervisedLearner
from toolkit.matrix import Matrix
import numpy as np
import matplotlib as plt


class BackProp(SupervisedLearner):
    """
    A Multi-layered Neural Network using backpropagation
    """


    def __init__(self, nodes_per_layer=3, hidden_layers=1, output_nodes=1, momentum=0, batch_norm = False):
        if hidden_layers > 1:
            # add hidden layer weights only if more than one layer is chosen
            self.weights = np.random.rand(hidden_layers, nodes_per_layer + 1)
        else:
            self.weights = np.array([])

        # The following weights are for testbp.arff
        # bias weights are moved to the end
        # self.weights = [[[0.2, -0.1, 0.1], [0.3, -0.3, -0.2]],
        #                 [[-0.2, -0.3, 0.1], [-0.1, 0.3, 0.2]],
        #                 [[-0.1, 0.3, 0.2], [-0.2, -0.3, 0.1]]]

        # these weights are for testbp2.arff
        # ordering is 5, 4, 6
        #             8, 7, 9
        #             11, 10, 12
        self.input_layer = np.array([[-0.03, 0.03, -0.01], [0.04, -0.02, 0.01], [.03, 0.02, -0.02]])
        # order is w1, w2, w3, w0
        self.output_layer = np.array([[-0.01, 0.03, 0.02, 0.02]])

        # self.weights = np.array(self.weights)
        ## Note on Weights ##
        # Weights are organized backwards = the top of the array is the leftmost(bottom) layer
        self.output_layer: np.ndarray
        self.nodes_per_layer = nodes_per_layer
        self.hidden_layers = hidden_layers
        self.accuracy_hash = {}
        self.epoch_count = 0
        self.learning_rate: float = .175
        self.momentum: float = .1
        self.labels = []
        self.output_classes = 0
        self.batch_norm_enabled = batch_norm
        self.has_hidden = self.hidden_layers > 1

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

    def train(self, features, labels):
        """
        :type features: Matrix
        :type labels: Matrix
        """
        # only need n - 1 nodes
        #self.output_classes = len(set(labels.col(labels.cols - 1))) - 1
        # set up output layer => the number of classes by the size of the previous hidden layer
        #self.output_layer = np.random.rand(self.output_classes, self.nodes_per_layer + 1)
        # setup input layer to match specs - number of nodes to connect to, number of inputs plus bias
        #self.input_layer = np.random.rand(len(self.weights[0]), features.cols + 1)
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
            accuracy_for_epoch: float = self._measure_accuracy(self.predict(features_bias, labels,
                                                                            individual_pred=False), labels)
            self.accuracy_hash[str(self.epoch_count)] = accuracy_for_epoch
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

    def predict(self, features, labels, individual_pred=True):
        """
        :type features: [float]
        :type labels: [float]
        :type individual_pred: [bool]
        """
        if individual_pred:  # being called from manager.py
            return self._feed_forward(features.data)

        else:  # otherwise we passed in a matrix
            prediction_list = []
            for row in features.data:
                prediction_list.append(self._feed_forward(row))
            return prediction_list



    def _feed_forward(self, row):
        self.output_matrix = np.zeros(tuple([self.hidden_layers + 1, self.nodes_per_layer, self.nodes_per_layer + 1]))
        # send input as first input layer
        input_values = row
        next_input_values: list = []
        ##### Start with input layer #####
        output_row = []
        for index_node, row in enumerate(self.input_layer):
            net = np.dot(input_values, row)
            output = self.output_sigmoid(net)
            output_row.append(output)
            next_input_values.append(float(output))
        # add bias for hidden layers
        output_row.append(1)
        input_values = output_row
        self.input_outputs = np.array(output_row)

        #### Feed that into hidden layers ####
        self.hidden_outputs = []
        for index_matrix, weight_matrix in enumerate(self.weights):
            output_row = []
            for index_node, row in enumerate(weight_matrix):
                net = np.dot(input_values, row)
                output = self.output_sigmoid(net)
                output_row.append(output)
                next_input_values.append(float(output))
                if row == weight_matrix[-1]:
                    output_row.append(1)
                    input_values = output_row
                    self.hidden_outputs.append(output_row)

        ##### Feed hidden layer output into
        final_output = []
        for index_node, row in enumerate(self.output_layer):
            net = np.dot(input_values, row)
            output = self.output_sigmoid(net)
            final_output.append(output)
            next_input_values.append(float(output))


        return final_output


    @staticmethod
    def _delta_output(target, output):
        return (target - output) * output * (1 - output)

    @staticmethod
    def _delta_hidden(output, delta_previous, weight_matrix):
        # output (1- output) * sum of weights to next layer * delta of that next layer
        # remove bias from delta calculation
        weight_matrix = weight_matrix[:-1]
        deltas = weight_matrix * delta_previous
        # lastly compute f'(x)
        return  output * (1 - output) * deltas

    def _back_propagate(self, output, target, input, batch_norm, end_of_batch=False):
        # use np for array operations
        output = np.array(output)
        target = np.array(target)
        input = np.array(input)
        matrix_num = len(self.weights)


        #### Delta for Output Layer ####
        output_deltas = self._delta_output(target, output)
        last_used_weights = self.output_layer[-1]

        ## Deltas for Hidden Layers ##
        delta_prev = output_deltas
        hidden_deltas = np.zeros_like(self.weights)
        # go back through each hidden layer
        for iteration, layer in enumerate(self.weights):
            output_val = self.hidden_outputs[len(self.hidden_outputs - 1 - iteration)]
            last_used_weights = self.weights[matrix_num - iteration]
            hidden_delta = self._delta_hidden(output_val,
                                           delta_prev, last_used_weights)
            # reference not the last, but the one before it and then iterate downward
            hidden_deltas[iteration] = hidden_delta
            # grab the last row
            delta_prev = hidden_delta

        ## Deltas for Input Layer ##
        input_deltas: list
        input_deltas = self._delta_hidden(output=self.input_outputs[:-1],
                                          delta_previous=delta_prev, weight_matrix=last_used_weights)


        print("The error values are: \n {} \n {} \n {}".format(output_deltas, hidden_deltas, input_deltas))

        ###### Configure outputs ######
        if self.has_hidden:
            output_storage = [self.hidden_outputs, self.input_outputs, input]
        else:
            output_storage = [self.input_outputs, input]

        ### prepare weight changes ###
        change_in_output_weights = self.learning_rate * output_storage[0] * output_deltas
        if self.hidden_outputs:
            change_in_hidden_weights = self.learning_rate * output_storage[1] * hidden_deltas
        else:
            change_in_hidden_weights = 0

        change_in_input_weights = []
        for value in input_deltas:
            change_in_input_weights.append(output_storage[-1] * value * self.learning_rate)

        # want to multiple each delta by each all the elements in inputs/outputs
        print("Done with backprop")
        print("Weights are now \n {} \n {} \n {} \n".format(self.input_layer + change_in_input_weights,
                                                            self.hidden_layers + change_in_hidden_weights,
                                                            self.output_layer + change_in_output_weights))

        ##### Update the weights #####
        self.input_layer += change_in_input_weights
        self.hidden_layers += change_in_hidden_weights
        self.output_layer += change_in_output_weights


