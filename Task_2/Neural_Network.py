import numpy as np
import math
from Task_2 import MeanVarNormalize
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
        self.bias = [np.random.uniform(-1, 1, (y, 1)) for y in n_units_per_layer[1:]]
        self.activation = 'sigmoid' if activation is None else 'reLU'

    def __init_weights(self):
        """
        Initialize the weights for each layer of the network. The following notation is used:
        -i: indicates the current weight in the layer k-th layer
        -j: indicates which neuron (in the k+1-th layer) the weight is connected to.
        :return:
        weights [[ndarray]]: list of matrices of the different weights in the hidden layers.
        """
        weights = [np.random.uniform(-1, 1, (j, i)) for i, j in zip(self.n_units_per_layer[:-1],
                                                                    self.n_units_per_layer[1:])]
        for i, weight_matrix in enumerate(weights):
            normalizer = MeanVarNormalize.MeanVarNormalize(weight_matrix)
            mean = normalizer.compute_mean()
            normalizer.compute_std(mean)
            weights[i] = normalizer.standardization()
        return weights

    def __normalize_weights(self, weights):
        """
        Peform the z-normalization of the generated weights.
        :param weights: the matrix of weights to be normalized
        :return: weights: the matrix of normalized weights
        """
        print(weights.shape)

        return weights

    def fit(self, X, y):
        """
        The training function for the model. The function fits the data using forward and backpropagation algorithm.
        :param X: the input data. It contains all the vectors of pixel for each image in the training set
        :param y: the truth values
        """
        for pixels_matrix in X:
            n = pixels_matrix.shape[0] * pixels_matrix.shape[1]
            pixels_vector = np.reshape(pixels_matrix, (n, 1))
            self.__forward_propagation(pixels_vector)
            self.__backpropagation()

    # noinspection PyShadowingNames
    def __apply_activation(self, z):
        """
        The function used to apply the activation function for each neuron. The available functions are
        sigmoid and reLU.
        :param z: vector of all the linear combination (W*a+b) for each neuron in a particular hidden layer.
        :return: a vector of the activation values computed by the corresponding function.
        """
        if self.activation == 'sigmoid':
            return 1/(1+np.exp(-z))
        else:
            return np.max(0, z)

    def __forward_propagation(self, a):
        """
        Apply feedforward propagation algorithm. This method simply computes the values for the neurons in the next
        hidden layer. For this purpose, the following formula is applied: a^(i+1) = act_func(z^(i))
        :param: a is the vector that contains the values for the neurons in the previous layer.
        :return: returns the same vector a given in input with the new values computed by the activation function.
        """
        print('perform forward propagation')
        for weights, bias in zip(self.weights, self.bias):
            z = np.dot(weights, a) + bias
            a = self.__apply_activation(z)
        return a

    def __backpropagation(self):
        print('perform backpropagation')


net = NeuralNetwork([784, 5, 5, 10])
dataset = Dataset.Dataset()
X = dataset.train_data
print(X.shape)
y = dataset.test_data
net.fit(X, y)
