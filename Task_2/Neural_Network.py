from sklearn.metrics import accuracy_score
import time
import torch as th
from torchvision.datasets import MNIST
import numpy as np

class Dataset:
    """
    The class that wraps the MNIST dataset downloaded and loaded by pytorch-vision.
    Each digit is represented as a vector of 28x28 pixels, where each pixel can have values between 0-255.
    In total there are 60000 elements in the training set while the elements in the test set are 10000.
    """
    def __init__(self):
        self.__train_loader = MNIST(root='.', download=True)
        self.__test_loader = MNIST(root='.', train=False)

    def get_train_data(self, perc):
        size = int(60000 * perc)
        X = self.__train_loader.train_data.numpy()
        X = np.reshape([(x/255) for x in X[:size]], (784, size))
        y = self.__train_loader.train_labels.numpy()
        y = self.__one_hot_encoding(y[:size])
        return X, y

    def get_test_data(self, perc):
        size = int(10000 * perc)
        X = self.__test_loader.test_data.numpy()
        X = np.reshape([(x / 255) for x in X[:size]], (784, size))
        y = self.__test_loader.test_labels.numpy()
        y = self.__one_hot_encoding(y[:size])
        return X, y

    def __one_hot_encoding(self, y):
        vectorized_labels = np.zeros((10, y.shape[0]))
        for vector, digit in zip(vectorized_labels.transpose(), y):
            vector[digit] = 1
        return vectorized_labels

    @property
    def train_data(self):
        return self.__train_data

    @property
    def train_labels(self):
        return self.__train_labels

    @property
    def test_data(self):
        return self.__test_data

    @property
    def test_labels(self):
        return self.__test_labels

    def get_item(self, index):
        return self.__train_data[index]


class Layer:

    def __init__(self, n_neurons, input_layer=False, activation=None):
        self.n_neurons = n_neurons
        self.input_layer = input_layer
        self.activation = activation


class CategoricalCrossEntropy:

    @staticmethod
    def delta(y_pred, y):
        result = y_pred - y
        return result

    @staticmethod
    def compute_loss(y_pred, y):
        N = y.shape[1]
        maxes = np.maximum(y_pred, 1e-15)
        logs = np.log(maxes)
        error = -np.sum(y * logs)/N
        return error


class Softmax:

    def apply_activation(self, z):
        num = z - np.max(z, axis=0, keepdims=True)
        num = np.exp(num)
        result = num / np.sum(num, axis=0, keepdims=True)
        return result

class Sigmoid:

    def apply_activation(self, z):
        return 1 / (1 + np.exp(-z))

    def derivative(self, z):
        return self.apply_activation(z)*(1-self.apply_activation(z))

class ReLU:

    def apply_activation(self, z):
        a = np.maximum(0, [x for x in z])
        return a

    def derivative(self, z):
        z_prime = np.zeros(z.shape)
        for i in range(z.shape[0]):
            for j in range(z.shape[1]):
                z_prime[i, j] = 0 if z[i, j] < 0 else 1
        return z_prime

# class LeakyReLU:
#     def apply_activation(self, z):
#         a = np.maximum(0.1, [x for x in z])
#         return a
#
#     def derivative(self, z):
#         z_prime = np.zeros(z.shape)
#         for i in range(z.shape[0]):
#             for j in range(z.shape[1]):
#                 z_prime[i, j] = 0.1 if z[i, j] < 0 else 1
#         return z_prime


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

    def compile(self, loss):
        self.weights.append(np.zeros((1, 1)))
        self.bias.append(np.zeros((1, 1)))
        # append a fist element as a zero because the activation function can be applied only to the hidden layers
        # thus in the Z values for the nodes in the input layer is set to zero and it will never be modified
        self.Z.append(np.zeros((1, 1)))
        self.A.append(np.zeros((1, 1)))
        for l in range(len(self.layers)-1):
            w = np.array(np.random.uniform(-1, 1, (self.layers[l+1].n_neurons, self.layers[l].n_neurons))
                         * np.sqrt(2/self.layers[l].n_neurons))
            b = np.array(np.random.uniform(-1, 1, (self.layers[l+1].n_neurons, 1)))
            self.weights.append(w)
            self.bias.append(b)
        self.loss = loss

    def fit(self, X_train, y_train, epochs=100, lr=0.09, lamda=0.01, min_delta=1e-5, patience=5):
        """
        The training function for the model. The weights and biases are updated using the SGD with mini batches.
        The gradient is computed using the backpropagation technique.
        :param min_delta:
        :param lr:
        :param lamda:
        :param X_train: the input data. It contains all the vectors of pixel for each image in the training set
        :param y_train: the truth values
        :param batch_size: the size of each batch
        :param epochs: the number of epochs or iteration required for the whole training process
        """
        loss_history = []
        start_time = time.process_time()
        for e in range(epochs):
            print(f'EPOCH # {e}')
            output = self.forward(X_train)
            loss_value = self.loss.compute_loss(output, y_train)
            print(f'LOSS: {loss_value}')
            loss_history.append(loss_value)
            grad_weights, grad_biases = self.backward(output, y_train)
            for l in range(len(self.weights)-1, 0, -1):
                self.weights[l] -= ((lr * (grad_weights[l-1])) - (lr*lamda)/X_train.shape[1])
                self.bias[l] -= (lr*grad_biases[l-1])
            if e > patience:
                stop_counter = 0
                for previous_loss in loss_history[-1:-patience-1:-1]:
                    if round(previous_loss, 6) < round(loss_value, 6):
                        stop_counter += 1
                if stop_counter >= patience:
                    print(f'Early Stopping. Actual loss: {loss_value}.')
                    break
        print(f'Time to finish: {round((time.process_time() - start_time), 3)}')

    def forward(self, X):
        a_l = X
        A = []
        Z = []
        A.append(a_l)
        for l in range(1, NeuralNetwork.n_layers):
            z_l = np.dot(self.weights[l], a_l) + self.bias[l]
            a_l = self.layers[l].activation.apply_activation(z_l)
            Z.append(z_l)
            A.append(a_l)

        self.A[1:] = A[1:]
        self.Z[1:] = Z
        return a_l

    def backward(self, output, y):
        d_weights, d_biases = [np.zeros((1, 1))], [np.zeros((1, 1))]
        delta = self.loss.delta(output, y)
        db = 1/y.shape[1] * np.sum(delta, axis=1, keepdims=True)
        d_biases.append(db)
        dw = np.dot(delta, self.A[-2].transpose()) * 1/y.shape[1]
        d_weights.append(dw)
        for l in range(NeuralNetwork.n_layers-2, 1, -1):
            l_prev, l_current, l_next = l-1, l, l+1
            delta = np.dot(self.weights[l_next].transpose(), delta) * self.layers[l_current].\
                activation.derivative(self.Z[l_current])
            dw = 1/y.shape[1] * np.dot(delta, self.A[l_prev].transpose())
            d_weights.append(dw)
            db = 1/y.shape[1] * np.sum(delta, axis=1, keepdims=True)
            d_biases.append(db)
        d_weights = [x for x in d_weights[1:]]
        d_biases = [x for x in d_biases[1:]]
        d_weights.reverse()
        d_biases.reverse()
        d_weights.insert(0, np.zeros((1, 1)))
        d_biases.insert(0, np.zeros((1, 1)))
        return d_weights, d_biases

    def predict(self, X_test, y_test):
        # get the outputs
        output = self.forward(X_test)
        y_pred = np.argmax(output, axis=0)
        y_true = np.argmax(y_test, axis=0)
        correct = 0
        for pred, true in zip(y_pred, y_true):
            correct += 1 if pred == true else 0
        errors = y_test.shape[1] - correct
        score = correct / y_test.shape[1]
        print(f'Accuracy {score}')
        print(f'Correctly classified digits: {correct}')
        print(f'Misclassified digits: {errors}')


class StochasticGradientDescent(NeuralNetwork):

    def __init__(self):
        super().__init__()

    def fit(self, X_train, y_train, epochs=100, lr=0.1, batch_size=500, lamda=1, patience=5):
        start_time = time.process_time()
        epoch_history = []
        N = X_train.shape[1]
        for e in range(epochs):
            t = X_train.shape[1] / batch_size
            random_indices = np.random.permutation(X_train.shape[1])
            X_shuffled = X_train[:, random_indices]
            y_shuffled = y_train[:, random_indices]
            batch_history = []
            for i in range(int(t)):
                x_batch = X_shuffled[:, i:i+X_shuffled.shape[1]:batch_size]
                y_batch = y_shuffled[:, i:i+y_shuffled.shape[1]:batch_size]
                output = self.forward(x_batch)
                loss_value = self.loss.compute_loss(output, y_batch)
                batch_history.append(loss_value)
                grad_weights, grad_bias = self.backward(output, y_batch)
                for l in range(len(self.weights) - 1, 0, -1):
                    self.weights[l] -= ((lr * (grad_weights[l-1])) - (lr*lamda)/batch_size)
                    self.bias[l] -= (lr * grad_bias[l-1])
            avg_loss = np.sum(batch_history) / N
            if e > patience:
                stop_counter = 0
                for previous_loss in epoch_history[-1:-patience-1:-1]:
                    if round(previous_loss, 6) < round(avg_loss, 6):
                        stop_counter += 1
                if stop_counter >= patience:
                    print(f'Early Stopping. Actual loss: {round(avg_loss, 6)}.')
                    break
            epoch_history.append(avg_loss)
            print(f'EPOCH # {e}')
            print(f'LOSS: {avg_loss}')

        print(f'Time to finish: {round((time.process_time() - start_time), 3)}')


dataset = Dataset()
X_train, y_train = dataset.get_train_data(perc=0.5)
X_test, y_test = dataset.get_test_data(perc=0.5)

net = StochasticGradientDescent()
net.add_layer(Layer(784, input_layer=True))
net.add_layer(Layer(128, activation=ReLU()))
net.add_layer(Layer(128, activation=ReLU()))
net.add_layer(Layer(128, activation=ReLU()))
net.add_layer(Layer(10, activation=Softmax()))
net.compile(loss=CategoricalCrossEntropy())
net.fit(X_train, y_train, epochs=2000, batch_size=100, lr=0.01, lamda=0.0001, patience=5)
net.predict(X_test, y_test)
