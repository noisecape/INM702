import numpy as np
from Task_2 import Dataset


class Layer:

    def __init__(self, n_neurons, input_layer=False, activation=None):
        self.n_neurons = n_neurons
        self.input_layer = input_layer
        self.activation = activation


class CrossEntropy:

    def delta(self, y_pred, y):
        return y_pred - y

    def compute_loss(self, y_pred, y):
        return -np.mean(y * np.log(y_pred) + (1.0-y) * np.log(1.0 - y_pred))


class Sigmoid:

    def apply_activation(self, z):
        return 1 / (1 + np.exp(-z))

    def derivative(self, z):
        return self.apply_activation(z)*(1-self.apply_activation(z))

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
        self.activations = []
        self.z = []
        self.weights = []
        self.bias = []

    def add_layer(self, layer):
        self.layers.append(layer)
        NeuralNetwork.n_layers += 1

    def compile(self, loss, optimizer=None, lr=0.001):
        self.weights.append(np.zeros((1, 1)))
        self.bias.append(np.zeros((1, 1)))
        self.z.append(np.zeros((1, 1)))
        self.activations.append(np.zeros((1, 1)))
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


    def fit(self, X_train, y_train, epochs=100, lr=0.001):
        """
        The training function for the model. The weights and biases are updated using the SGD with mini batches.
        The gradient is computed using the backpropagation technique.
        :param X_train: the input data. It contains all the vectors of pixel for each image in the training set
        :param y_train: the truth values
        :param batch_size: the size of each batch
        :param epochs: the number of epochs or iteration required for the whole training process
        """
        X_train.shape
        y_train.shape
        for x_sample, y_label in zip(X_train, y_train):
            loss_history = []
            for e in range(epochs):
                #forward
                self.__forward(x_sample)
                #backward
                grad_weights, grad_biases = self.__backward(self.activations, y_label)
                for l in range(len(self.weights)-1, 0, -1):
                    self.weights[l] -= (lr*np.sum(grad_weights[l]))
                    self.bias[l] -= (lr*np.sum(grad_biases[l]))
                self.__forward(x_sample)
                output = self.activations[-1]
                loss_value = self.loss.compute_loss(output, y_train)
                loss_history.append(loss_value)
                #update weights
                #check loss value
            print(loss_history)

    def __backward(self, activations, y_sample):
        d_weights, d_biases = [np.zeros((1, 1))], [np.zeros((1, 1))]
        delta = self.loss.delta(activations[-1], y_sample)
        d_biases.append(delta)
        d_weights.append(np.dot(activations[-2], delta.transpose()))
        for l in range(NeuralNetwork.n_layers-2, 0, -1):
            l_prev, l_current, l_next = l-1, l, l+1
            dw = np.dot(self.weights[l_next].transpose(), delta)
            delta = self.layers[l_current].activation.derivative(self.z[l_current]) * dw
            d_weights.append(dw)
            d_biases.append(delta)
        d_weights = [x for x in d_weights[1:]]
        d_biases = [x for x in d_biases[1:]]
        d_weights.reverse()
        d_biases.reverse()
        d_weights.insert(0, np.zeros((1, 1)))
        d_biases.insert(0, np.zeros((1, 1)))
        return d_weights, d_biases

    def __forward(self, x):
        a_prev = x
        for weight, bias, layer in zip(self.weights[1:], self.bias[1:], self.layers[1:]):
            z = np.dot(weight, a_prev) + bias
            self.z.append(z)
            a_prev = layer.activation.apply_activation(z)
            self.activations.append(a_prev)

    def __update_weights_biases(self, batch, batch_size):
        """
        Function used to update values for the weights and biases. To achieve that, backpropagation algorithm is used.
        To update the weights it is necessary to store biases and weights at each level of the network.
        :param batch: the data currently used to train the network
        """
        for x_batch, y_batch in batch:
            y_batch = np.reshape(y_batch, (-1, 1))
            gradients_weights, gradients_biases = self.__back_propagation(x_batch, y_batch)
            for layer in range(len(self.weights)-1, 0, -1):
                self.weights[layer] = self.weights[layer] - (self.learning_rate/batch_size)*gradients_weights[layer]\
                    .transpose()
                self.bias[layer] = self.bias[layer] - (self.learning_rate/batch_size)*gradients_biases[layer]

    def __back_propagation(self, X_batch, y_batch):
        """
        Apply backpropagation to compute the gradients for both the weights and the biases. It is necessary to keep
        track of the z values for each neuron at each particular layer to compute the activations value for each neuron.
        :param X_batch: the input data.
        :param y_batch: the truth value for the corresponding input data.
        :return: gradients_weights, gradients_biases: two lists which stores the gradients weights and the gradients
        biases respectively.
        """
        gradients_weights = []
        gradients_biases = []
        # FIRST APPLY FORWARD PROPAGATION AND STORE the a-s and z-s
        a_values, z_values = self.__forward_propagation(X_batch)
        # THEN, COMPUTE ERROR FOR THE LAST LAYER
        last_output = a_values[-1]
        delta = last_output - y_batch
        # THEN, APPEND DELTA IN THE LAST POSITION OF THE GRADIENT_BIAS
        gradients_biases.append(delta)
        # COMPUTE THE GRADIENT FOR THE LAST WEIGHTS AND ADD IT TO THE GRADIENT VECTOR
        gradients_weights.append(np.dot(a_values[-2], delta.transpose()))
        # START THE BACKPROP.
        for layer in range(len(self.weights)-1, 0, -1):
            next_layer = layer
            current_layer = layer-1
            prev_layer = current_layer-1
            delta = np.dot(self.weights[next_layer].transpose(), delta) * self.__der_act_function(z_values[current_layer])
            grad_w = np.dot(delta, a_values[prev_layer+1].transpose())
            gradients_biases.append(delta)
            gradients_weights.append(grad_w)
        gradients_weights.reverse()
        gradients_biases.reverse()
        # RETURN THE TWO VECTORS OF GRADIENTS
        return gradients_weights, gradients_biases

    def __der_act_function(self, z):
        if self.activation == 'sigmoid':
            return self.__apply_activation(z) * (1-self.__apply_activation(z))
        else:
            "compute derivative of reLU"

dataset = Dataset.Dataset()
X_train = dataset.debug_train_data
y_train = dataset.debug_train_labels
X_test = dataset.debug_test_data
y_test = dataset.debug_test_labels
net = NeuralNetwork()
net.add_layer(Layer(784, input_layer=True, activation='sigmoid')) # input layer
net.add_layer(Layer(30, activation=Sigmoid())) # hidden layer
net.add_layer(Layer(30, activation=Sigmoid())) # hidden layer
net.add_layer(Layer(10, activation=Sigmoid())) # output layer
net.compile(loss=CrossEntropy())
net.fit(X_train, y_train, epochs=5)
