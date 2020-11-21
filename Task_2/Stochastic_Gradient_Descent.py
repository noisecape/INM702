import numpy as np


class StochasticGradientDescent:

    def __init__(self, learning_rate):
        self.__learning_rate = learning_rate

    @property
    def learning_rate(self):
        return self.__learning_rate

    def fit(self, X, y):
        """
        Computes the gradient descent using the cross entropy
        :param X:
        :param y:
        :return:
        """
