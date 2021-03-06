from toolkit.supervised_learner import SupervisedLearner
from toolkit.matrix import Matrix
import numpy as np
import matplotlib as plt
import operator



class BackProp(SupervisedLearner):
    """
    A Multi-layered Neural Network using backpropagation
    """


    def __init__(self, nodes_per_layer=32, num_hidden_layers=4, momentum=.05, learning_rate=.1,
                 batch_norm=False, validation_set=True):
        self.validation_set = validation_set
        self.min = -.05
        self.max = .05
        self.output_layer = np.ndarray
        if num_hidden_layers > 1:
            # add hidden layer weights only if more than one layer is chosen
            self.hidden_layers = np.random.uniform(low=-self.min, high=self.max,
                                                   size=(num_hidden_layers, nodes_per_layer, nodes_per_layer + 1))
        else:
            self.hidden_layers = np.array([])

        # The following weights are for testbp.arff
        # bias weights are moved to the end
        # self.input_layer = [[0.2, -0.1, 0.1], [0.3, -0.3, -0.2]]
        # self.weights = [[[-0.2, -0.3, 0.1], [-0.1, 0.3, 0.2]]]
        # self.output_layer =  [[-0.1, 0.3, 0.2], [-0.2, -0.3, 0.1]]

        ## Note on Weights ##
        # Weights are organized backwards = the top of the array is the leftmost(bottom) layer
        # these weights are for testbp2.arff
        # ordering is 5, 4, 6
        #             8, 7, 9
        #             11, 10, 12
        # self.input_layer = np.array([[-0.03, 0.03, -0.01], [0.04, -0.02, 0.01], [.03, 0.02, -0.02]])
        # # # order is w1, w2, w3, w0
        # # self.weights = []
        # self.output_layer = np.array([[-0.01, 0.03, 0.02, 0.02]])
        # self.weights = np.array(self.weights)

        self.accuracy_hash_train = {}
        self.nodes_per_layer = nodes_per_layer
        self.num_hidden_layers = num_hidden_layers
        self.accuracy_hash = {}
        self.epoch_count = 0
        self.learning_rate: float = learning_rate
        self.momentum: float = momentum
        self.last_change_in_outputs = 0
        self.last_change_in_hidden = 0
        self.last_change_in_inputs = 0
        self.labels = []
        self.output_classes = 0
        self.batch_norm_enabled = batch_norm
        self.has_hidden = num_hidden_layers > 1
        self.MSE = True
        self.model_type = "BackProp"
        self.learn_until = 50

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
        print("The learning rate for this model is {} \n with momentum {}".format(self.learning_rate,
                                                                                  self.momentum))
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




        ##### Setup Weights #####
        self.output_classes = len(set(labels.col(labels.cols - 1)))
        # set up output layer => the number of classes by the size of the previous hidden layer
        self.output_layer = np.random.uniform(low=self.min, high=self.max,
                                              size=(self.output_classes, self.nodes_per_layer + 1))
        # setup input layer to match specs - number of nodes to connect to, number of inputs plus bias
        self.input_layer = np.random.uniform(low=self.min, high=self.max,
                                             size=(self.nodes_per_layer, features_bias.cols + 1))
        # setup output layer to match specs
        self.num_output_layers = labels.cols
        # create a new features set with a bias for training
        last_row_num = features_bias.rows - 1
        self._is_nominal_output = labels.value_count(0) != 0

        self.best_inputs = self.input_layer
        self.best_hidden = self.hidden_layers
        self.best_output = self.output_layer

        if not self.validation_set:
            train_features = features_bias
            test_features = features_bias
            train_labels = labels
            test_labels = labels

        # start learning
        while self._is_still_learning(self):
            print(" #### On Epoch Number {}".format(self.epoch_count))
            train_features.shuffle(buddy=train_labels)
            for row_num in range(train_features.rows):
                #print("On input number {} #############".format(row_num + 1))
                row = train_features.row(row_num)
                #print("Feed Forward with row: {}".format(row))
                output = self._feed_forward(row)
                #print("backpropogating errors with output: {}".format(output))
                self._back_propagate(output, train_labels.row(row_num), row, self.batch_norm_enabled,
                                     (self.batch_norm_enabled and row_num == last_row_num))
            # test on validation set
            accuracy_for_epoch: float = self.measure_accuracy(test_features, test_labels, MSE=self.MSE)
            accuracy_for_epoch_train: float = self.measure_accuracy(train_features, train_labels, MSE=self.MSE)
            if self.epoch_count > 1:
                if (self.MSE and min(self.accuracy_hash.items(), key=lambda x: x[1])[1] < accuracy_for_epoch) or (not self.MSE and max(self.accuracy_hash.items(), key=lambda x: x[1])[1] < accuracy_for_epoch):
                    self.best_inputs = self.input_layer
                    self.best_hidden = self.hidden_layers
                    self.best_output = self.output_layer
            self.accuracy_hash[self.epoch_count] = accuracy_for_epoch
            self.accuracy_hash_train[self.epoch_count] = accuracy_for_epoch_train

            self.epoch_count += 1

        print("The best accuracy on the validation set was {}".format(
            max(self.accuracy_hash.items(), key = lambda x: x[1])))

        self.input_layer = self.best_inputs
        self.hidden_layers = self.best_hidden
        self.output_layer = self.best_output

        final_vs: float = self.measure_accuracy(test_features, test_labels, MSE=self.MSE)
        final_ts: float = self.measure_accuracy(train_features, train_labels, MSE=self.MSE)
        print("The final best for VS: {} and for TS: {}".format(final_vs, final_ts))

        return

    @staticmethod
    def _is_still_learning(self):
        margin_increasing = .005
        if len(self.accuracy_hash) <= self.learn_until:
            return True
        # check that the accuracy hasn't improved for 5 iterations
        baseline: float = self.accuracy_hash[self.epoch_count - self.learn_until]
        # check the end point also
        for iteration in range(self.epoch_count - self.learn_until, self.epoch_count):
            if self.MSE:
                if self.accuracy_hash[iteration] - baseline < -1 * margin_increasing:
                    return True
            else:
                if self.accuracy_hash[iteration] - baseline > margin_increasing:
                    return True
        else:
            return False

    @staticmethod
    def output_sigmoid(net):
        return net if net > 0 else 0

    def predict(self, features, labels, individual_pred=True):
        """
        :type features: [float]
        :type labels: [float]
        :type individual_pred: [bool]
        """
        if individual_pred:  # being called from manager.py
            features.append(1)
            return self._feed_forward(features)

        else:  # otherwise we passed in a matrix
            prediction_list = []
            for row in features.data:
                prediction_list.append(self._feed_forward(row))
            return prediction_list



    def _feed_forward(self, row):
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

        for index_matrix, matrix in enumerate(self.hidden_layers):
            output_row = []
            for index_row, row in enumerate(matrix):
                net = np.dot(input_values, row)
                output = self.output_sigmoid(net)
                output_row.append(output)
                if index_row == matrix.shape[0] - 1:
                    output_row.append(1)
                    input_values = output_row
                    self.hidden_outputs.append(output_row)
        self.hidden_outputs = np.array(self.hidden_outputs)

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

    def _delta_hidden_try(self, output, delta_previous, weight_matrix):
        # output (1- output) * sum of weights to next layer * delta of that next layer
        # get the layer before delta values
        # get the layer before delta
        deltas = []
        for index, row in enumerate(weight_matrix[:, :-1].T):
            deltas.append(np.dot(row, delta_previous))
        deltas = np.array(deltas)
        return output * (1 - output) * deltas

    @staticmethod
    def _delta_hidden(output, delta_previous, weight_matrix):
        # output (1- output) * sum of weights to next layer * delta of that next layer
        # remove bias from delta calculation
        deltas = np.dot(weight_matrix, delta_previous)
        # lastly compute f'(x)
        return  output * (1 - output) * deltas

    def _back_propagate(self, output, target, input, batch_norm, end_of_batch=False):
        # use np for array operations
        output = np.array(output)
        target = self._make_target(target)
        input = np.array(input)
        matrix_num = len(self.hidden_layers)

        #### Delta for Output Layer ####
        output_deltas = np.array(self._delta_output(target, output))
        next_to_use_weights = self.output_layer
        # convert to numpy

        ## Deltas for Hidden Layers ##
        delta_prev = output_deltas
        # deltas don't include bias node
        hidden_deltas = []
        # go back through each hidden layer
        if self.has_hidden:
            for iteration in range(self.num_hidden_layers):
                output_val = self.hidden_outputs[len(self.hidden_outputs) - 1 - iteration]
                hidden_delta = self._delta_hidden_try(output_val[:-1],
                                               delta_prev, next_to_use_weights)
                # reference not the last, but the one before it and then iterate downward
                hidden_deltas.append(hidden_delta)
                next_to_use_weights = self.hidden_layers[len(self.hidden_layers) - 1 - iteration]
                # grab the last row
                delta_prev = hidden_delta
            # convert to numpy
            hidden_deltas = np.array(hidden_deltas)

        ## Deltas for Input Layer ##
        input_deltas: list
        input_deltas = self._delta_hidden_try(output=self.input_outputs[:-1],
                                          delta_previous=delta_prev, weight_matrix=next_to_use_weights)


        #print("The error values are: \n {} \n {} \n {}".format(output_deltas, hidden_deltas, input_deltas))

        ###### Configure outputs ######
        if self.has_hidden:
            output_storage = np.array([self.hidden_outputs[0],
                              # combine these two for the loop
                              np.vstack([self.hidden_outputs[1:],
                                         self.input_outputs.reshape(1, self.input_outputs.shape[0])]),
                              input])
        else:
            output_storage = [self.input_outputs, input]
        # TODO: with the way we get the output from storage for hidden layers
        ### prepare weight changes ###
        change_in_output_weights = []
        for delta in output_deltas:
            change_in_output_weights.append(np.array(output_storage[0]) * delta * self.learning_rate)
        # concatenate to remove extra list structure
        change_in_output_weights = np.array(np.array(change_in_output_weights))

        if self.has_hidden:
            change_in_hidden_weights = []
            for index, delta_values in enumerate(hidden_deltas):
                    delta_mul = delta_values.reshape(len(delta_values), 1)
                    current_output = output_storage[1][index].reshape(1, len(output_storage[1][index]))
                    change_in_hidden_weights.append(delta_mul * current_output * self.learning_rate)
            # change_in_hidden_weights = np.dot(hidden_deltas.T, np.array(output_storage[1])) * self.learning_rate
        else:
            change_in_hidden_weights = 0
        change_in_hidden_weights = np.array(change_in_hidden_weights)


        change_in_input_weights = []
        for delta in input_deltas:
            change_in_input_weights.append(output_storage[-1] * delta * self.learning_rate)
        change_in_input_weights = np.array(change_in_input_weights)


        ##### Do momentum ####
        change_in_input_weights = np.add(change_in_input_weights,
                                         self.momentum * self.last_change_in_inputs,
                                         out=change_in_input_weights, casting="unsafe")
        change_in_hidden_weights = np.add(change_in_hidden_weights,
                                         self.momentum * self.last_change_in_hidden,
                                         out=change_in_hidden_weights, casting="unsafe")
        change_in_output_weights = np.add(change_in_output_weights,
                                         self.momentum * self.last_change_in_outputs,
                                         out=change_in_output_weights, casting="unsafe")


        ##### Update the weights #####
        self.input_layer += change_in_input_weights
        self.last_change_in_inputs = change_in_input_weights
        self.hidden_layers += change_in_hidden_weights
        self.last_change_in_hidden = change_in_hidden_weights
        self.output_layer += change_in_output_weights
        self.last_change_in_outputs = change_in_output_weights

        # print("Change in weights (including momentum) are \n {} \n {} \n {} \n".format(change_in_input_weights,
        #                                                           change_in_hidden_weights,
        #                                                           change_in_output_weights))
        # print("Done with backprop")
        # print("Weights are now \n {} \n {} \n {} \n".format(self.input_layer,
        #                                                     self.hidden_layers,
        #                                                     self.output_layer))

    def measure_accuracy(self, features, labels, confusion=None, MSE=None):
        return super(BackProp, self).measure_accuracy(features, labels, MSE=MSE)

    def _make_target(self, target):
        """

        :param target: the value of the class given
        :return:
        """
        if self._is_nominal_output:
            if type(target) == list:
                target = target[0]
            target_array = np.zeros(len(self.output_layer))
            target_array[int(target)] = 1
            return np.array(target_array)
        else:
            return np.array(target)


