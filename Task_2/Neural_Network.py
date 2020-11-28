import numpy as np
from Task_2 import Dataset
import time

class Layer:

    def __init__(self, n_neurons, input_layer=False, activation=None):
        self.n_neurons = n_neurons
        self.input_layer = input_layer
        self.activation = activation


class CrossEntropy:

    @staticmethod
    def delta(y_pred, y):
        return y_pred - y

    @staticmethod
    def compute_loss(y_pred, y):
        return -np.mean(y * np.log(y_pred) + (1.0-y) * np.log(1.0 - y_pred))

class Sigmoid:

    def apply_activation(self, z):
        return 1 / (1 + np.exp(-z))

    def derivative(self, z):
        return self.apply_activation(z)*(1-self.apply_activation(z))

class Softmax:

    def apply_activation(self, z):
        return 1 / (1 + np.exp(-z))

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

    n_layers = 0

    def __init__(self):
        self.layers = []
        self.optimizer = None
        self.loss = None
        self.Z = []
        self.A = []
        self.weights = []
        self.bias = []

    def add_layer(self, layer):
        self.layers.append(layer)
        NeuralNetwork.n_layers += 1

    def compile(self, loss, optimizer=None, lr=0.0001):
        self.weights.append(np.zeros((1, 1)))
        self.bias.append(np.zeros((1, 1)))
        self.Z.append(np.zeros((1, 1)))
        self.A.append(np.zeros((1, 1)))
        for l in range(len(self.layers)-1):
            w = np.array(np.random.uniform(-1, 1, (self.layers[l+1].n_neurons, self.layers[l].n_neurons)))
            b = np.array(np.random.uniform(-1, 1, (self.layers[l+1].n_neurons, 1)))
            self.weights.append(w)
            self.bias.append(b)
        self.loss = loss

    def predict(self, X_test, y_test):
        print('predict values')
        for x, label in zip(X_test, y_test):
            outputs, _ = self.__forward_propagation(x)
            predicted_digit = max(outputs[-1])


    def fit(self, X_train, y_train, epochs=100, lr=0.01):
        """
        The training function for the model. The weights and biases are updated using the SGD with mini batches.
        The gradient is computed using the backpropagation technique.
        :param X_train: the input data. It contains all the vectors of pixel for each image in the training set
        :param y_train: the truth values
        :param batch_size: the size of each batch
        :param epochs: the number of epochs or iteration required for the whole training process
        """
        loss_history = np.array(np.zeros((epochs, 1)))
        partial_loss = 0
        for e in range(epochs):
            #forward
            output = self.__forward(X_train)
            #backward
            grad_weights, grad_biases = self.__backward(output, y_train)
            for l in range(len(self.weights)-1, 0, -1):
                self.weights[l] -= (lr*np.sum(grad_weights[l]))
                self.bias[l] -= (lr*np.sum(grad_biases[l]))
            output = self.__forward(X_train)
            loss_value = self.loss.compute_loss(output, y_train)
            loss_history[e] = loss_value
        print(f'EPOCH  #{e} COMPLETED, TRAIN SAMPLE #{len(loss_history)}')
        print(loss_history)
        print(time.process_time())

    def __forward(self, X):
        a_l = X
        for l in range(1, NeuralNetwork.n_layers):
            z_l = np.dot(self.weights[l], a_l) + self.bias[l]
            a_l = self.layers[l].activation.apply_activation(z_l)
            self.Z.append(z_l)
            self.A.append(a_l)
        return a_l

    def __backward(self, output, y):
        d_weights, d_biases = [np.zeros((1, 1))], [np.zeros((1, 1))]
        delta = self.loss.delta(output, y)
        d_biases.append(delta)
        dw = np.dot(delta, self.A[-2].transpose()) * 1/y.shape[1]
        d_weights.append(dw)
        for l in range(NeuralNetwork.n_layers-2, 0, -1):
            l_prev, l_current, l_next = l-1, l, l+1
            dw = np.dot(self.weights[l_next].transpose(), delta)
            delta = dw * self.layers[l_current].activation.derivative(self.Z[l_current])
            d_weights.append(dw)
            d_biases.append(delta)
        d_weights = [x for x in d_weights[1:]]
        d_biases = [x for x in d_biases[1:]]
        d_weights.reverse()
        d_biases.reverse()
        d_weights.insert(0, np.zeros((1, 1)))
        d_biases.insert(0, np.zeros((1, 1)))
        return d_weights, d_biases


dataset = Dataset.Dataset()
X_train = dataset.debug_train_data
y_train = dataset.debug_train_labels
X_test = dataset.debug_test_data
y_test = dataset.debug_test_labels
net = NeuralNetwork()
net.add_layer(Layer(784, input_layer=True, activation='sigmoid')) # input layer
net.add_layer(Layer(64, activation=Sigmoid()))
net.add_layer(Layer(32, activation=Sigmoid()))
net.add_layer(Layer(10, activation=Sigmoid()))
net.compile(loss=CrossEntropy())
net.fit(X_train, y_train, epochs=100, lr=0.001)