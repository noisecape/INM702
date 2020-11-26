import numpy as np
from Task_2 import MeanVarNormalize
from Task_2 import Dataset

print('im in the vectorization branch')

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

    def __init__(self, n_units_per_layer, activation=None, learning_rate=0.1):
        self.n_units_per_layer = n_units_per_layer
        self.weights = self.__init_weights()
        self.bias = [np.random.uniform(-1, 1, (y, 1)) for y in n_units_per_layer[1:]]
        self.activation = 'sigmoid' if activation is None else 'reLU'
        self.learning_rate = learning_rate

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

    def predict(self, X_test, y_test):
        print('predict values')
        for x, label in zip(X_test, y_test):
            outputs, _ = self.__forward_propagation(x)
            predicted_digit = max(outputs[-1])


    def fit(self, X_train, y_train, batch_size, epochs):
        """
        The training function for the model. The weights and biases are updated using the SGD with mini batches.
        The gradient is computed using the backpropagation technique.
        :param X_train: the input data. It contains all the vectors of pixel for each image in the training set
        :param y_train: the truth values
        :param batch_size: the size of each batch
        :param epochs: the number of epochs or iteration required for the whole training process
        """
        for e in range(epochs):
            training_data = [(x, y) for x, y in zip(X_train, y_train)]
            np.random.shuffle(training_data)
            mini_batches = [training_data[i:i+batch_size] for i in range(0, len(training_data), batch_size)]
            loss_history = []
            for counter, batch in enumerate(mini_batches):
                self.__update_weights_biases(batch, batch_size)
                # compute loss for this batch
                partial_losses = self.__compute_loss(batch)
                avg_loss = np.mean(partial_losses)
                loss_history.append(avg_loss)
                print(f'Loss value: {avg_loss}')
        print(f'############ Epochs: {e+1}###########')
        print('DONE')

    def __compute_loss(self, batch):
        partial_losses = []
        for train_sample in zip(batch):
            data = train_sample[0]
            X_train, y_train = data[0], data[1]
            y_train = np.reshape(y_train, (-1,1))
            outputs, _ = self.__forward_propagation(X_train)
            loss = self.__cross_entropy(outputs[-1], y_train)
            partial_losses.append(loss)
        return partial_losses

    @staticmethod
    def __cross_entropy(y_pred, y_train):
        return -y_train * np.log(y_pred) - (1.0-y_train) * np.log(1.0 - y_pred)

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

    def __forward_propagation(self, a):
        """
        Apply feedforward propagation algorithm. This method simply computes the values for the neurons in the next
        hidden layer. For this purpose, the following formula is applied: a^(i+1) = act_func(z^(i))
        :param: a is the vector that contains the values for the neurons in the previous layer.
        :return: a_values, z_values: two lists storing respectively the activation values and the z_values at each layer
        """
        a_values = []
        z_values = []
        for weights, bias in zip(self.weights, self.bias):
            z = np.dot(weights, a) + bias
            z_values.append(z)
            a = self.__apply_activation(z)
            a_values.append(a)
        return a_values, z_values

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


net = NeuralNetwork([784, 30, 10], learning_rate=0.001)
dataset = Dataset.Dataset()
X_train = dataset.debug_train_data
y_train = dataset.debug_train_labels
X_test = dataset.debug_test_data
y_test = dataset.debug_test_labels

net.fit(X_train, y_train, 30, 40)
net.predict(X_test, y_test)
