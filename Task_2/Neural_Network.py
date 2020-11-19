import numpy as np
import math
from Task_2 import Dataset

class NeuralNetwork:
    """
    The class for the implementation of the neural network. The class is fully parametrized in the sense that allows the
    user to set the number of total layer and number of neurons in the hidden layer. The user can also specify the type
    of activation function for the neurons in the hidden layer.
    Parameters:
        n_units_per_layer (list): specifies the number of units for each layer. The first element represents the number
        of inputs node. The last one the number of output neurons. The others represent the number of element in the
        hidden layers.
        weights (ndarray): list of matrices of the weights for each layer
        bias (ndarray): vector of biases.
        activation (str): by default is the sigmoid function. Possible values: sigmoid, reLU
    """

    def __init__(self, n_units_per_layer, activation=None):
        self.n_units_per_layer = n_units_per_layer
        self.weights = self.__init_weights()
        self.bias = np.random.randn(self.n_units_per_layer[-1], 1)
        self.activation = 'sigmoid' if activation is None else 'reLU'

    def __init_weights(self):
        """
        Initialize the weights for each layer of the network. The following notation is used:
        -i: indicates the current weight in the layer k-th layer
        -j: indicates which neuron (in the k+1-th layer) the weight is connected to.
        :return:
        weights [[ndarray]]: list of matrices of the different weights in the hidden layers.
        """
        weights = [np.random.randn(j, i) for i, j in zip(self.n_units_per_layer[:-1], self.n_units_per_layer[1:])]
        return weights

    def __apply_activation(self, z):
        if self.activation == 'sigmoid':
            return 1/(1+math.exp(z))
        else:
            return np.max(0, z)

    def __forward_propagation(self):
        print('perform forward propagation')


net = NeuralNetwork([3, 2, 1])
