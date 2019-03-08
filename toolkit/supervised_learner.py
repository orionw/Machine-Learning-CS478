from __future__ import (absolute_import, division, print_function, unicode_literals)

from .matrix import Matrix
import math
import numpy as np
import pandas as pd

# this is an abstract class


class SupervisedLearner:

    def train(self, features, labels):
        """
        Before you call this method, you need to divide your data
        into a feature matrix and a label matrix.
        :type features: Matrix
        :type labels: Matrix
        """
        raise NotImplementedError()

    def predict(self, features, labels):
        """
        A feature vector goes in. A label vector comes out. (Some supervised
        learning algorithms only support one-dimensional label vectors. Some
        support multi-dimensional label vectors.)
        :type features: [float]
        :type labels: [float]
        """
        raise NotImplementedError

    def measure_accuracy(self, features, labels, confusion=None, MSE=None):
        """
        The model must be trained before you call this method. If the label is nominal,
        it returns the predictive accuracy. If the label is continuous, it returns
        the root mean squared error (RMSE). If confusion is non-NULL, and the
        output label is nominal, then confusion will hold stats for a confusion matrix.
        :type features: Matrix
        :type labels: Matrix
        :type confusion: Matrix
        :rtype float
        """

        if features.rows != labels.rows:
            raise Exception("Expected the features and labels to have the same number of rows")
        if labels.cols != 1:
            raise Exception("Sorry, this method currently only supports one-dimensional labels")
        if features.rows == 0:
            raise Exception("Expected at least one row")

        backprop = False
        try:
            if self.model_type == "BackProp":
                backprop = True
                features_to_change = Matrix(features, 0, 0, features.rows, features.cols)
                features = features_to_change
        except AttributeError as e:
            # wasn't backprop, pass
            pass

        label_values_count = labels.value_count(0)
        if MSE or not label_values_count:
            print("MSE")
            # label is continuous
            pred = [0]
            sse = 0.0
            for i in range(features.rows):
                feat = features.row(i)
                targ = labels.row(i)
                # targ_real = np.zeros(self.output_classes)
                # targ_real[int(targ)] = 1
                # pred[0] = 0.0       # make sure the prediction is not biased by a previous prediction
                pred = self.predict(feat, pred)
                delta = targ[0] - pred[0]
                sse += delta**2
            return math.sqrt(sse / features.rows)

        else:
            # label is nominal, so measure predictive accuracy
            if confusion:
                confusion.set_size(label_values_count, label_values_count)
                confusion.attr_names = [labels.attr_value(0, i) for i in range(label_values_count)]

            correct_count = 0
            prediction = []
            print("There are {} rows".format(features.rows))
            for i in range(features.rows):
                if not i % 1000:
                    print("Done {} ###".format(i))
                feat = features.row(i)
                targ = int(labels.get(i, 0))
                if targ >= label_values_count:
                    raise Exception("The label is out of range")
                pred = self.predict(feat, prediction)
                if backprop and type(pred[0]) == np.float64:
                    # use a round/softmax
                    max_index = np.argmax(pred)
                    pred = [max_index if math.ceil(pred[max_index]) else ValueError]
                pred = (pred[0])
                # if confusion:
                #      confusion.set(targ, pred, confusion.get(targ, pred)+1)
                if type(pred) == pd.Series:
                    pred = int(pred[0])
                if pred == targ:
                    correct_count += 1

            return correct_count / features.rows


    def measure_accuracy_pandas(self, features, labels, confusion=None, MSE=None):
        """
        The model must be trained before you call this method. If the label is nominal,
        it returns the predictive accuracy. If the label is continuous, it returns
        the root mean squared error (RMSE). If confusion is non-NULL, and the
        output label is nominal, then confusion will hold stats for a confusion matrix.
        :type features: DataFrame
        :type labels: DataFrame
        :type confusion: Matrix
        :rtype float
        """

        if len(features) != len(labels):
            raise Exception("Expected the features and labels to have the same number of rows")
        if len(labels.columns) != 1:
            raise Exception("Sorry, this method currently only supports one-dimensional labels")
        if len(features) == 0:
            raise Exception("Expected at least one row")

        label_values_count = labels.nunique()[0]
        if MSE or not label_values_count:
            # label is continuous
            pred = [0]
            sse = 0.0
            for i in range(features.rows):
                feat = features.row(i)
                targ = labels.row(i)[0]
                targ_real = np.zeros(self.output_classes)
                targ_real[int(targ)] = 1
                pred[0] = 0.0       # make sure the prediction is not biased by a previous prediction
                pred = self.predict(feat, pred)
                delta = targ_real[0] - pred[0]
                sse += delta**2
            return math.sqrt(sse / features.rows)

        else:
            # label is nominal, so measure predictive accuracy
            if confusion:
                confusion.set_size(label_values_count, label_values_count)
                confusion.attr_names = [labels.attr_value(0, i) for i in range(label_values_count)]

            correct_count = 0
            prediction = []
            for i in range(len(features)):
                feat = features.iloc[i]
                targ = int(labels.iloc[i])
                if targ >= label_values_count:
                    raise Exception("The label is out of range")
                pred = self.predict(feat, prediction)
                pred = (pred[0])
                if type(pred) == pd.Series:
                    pred = int(pred[0])
                # if confusion:
                #      confusion.set(targ, pred, confusion.get(targ, pred)+1)
                if pred == targ:
                    correct_count += 1

            return correct_count / len(features)

