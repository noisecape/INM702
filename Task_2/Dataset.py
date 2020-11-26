from torchvision.datasets import MNIST
import numpy as np


class Dataset:
    """
    The class that wraps the MNIST dataset downloaded and loaded by pytorch-vision.
    Each digit is represented as a vector of 28x28 pixels, where each pixel can have values between 0-255.
    In total there are 60000 elements in the training set while the elements in the test set are 10000.
    """
    def __init__(self):
        train_loader = MNIST(root='.', download=True)
        test_loader = MNIST(root='.', train=False)

        # self.__train_data = self.__reshape_ditigs_matrix(train_loader.train_data.numpy())
        # self.__train_labels = self.__reshape_labels(train_loader.train_labels.numpy())
        # self.__test_data = self.__reshape_ditigs_matrix(test_loader.test_data.numpy())
        # self.__test_labels = self.__reshape_labels(test_loader.test_labels.numpy())

        # for debugging
        self.debug_train_data = self.__reshape_ditigs_matrix(train_loader.train_data.numpy(), 2000)
        self.debug_train_labels = self.__reshape_labels(train_loader.train_labels.numpy(), 2000)
        self.debug_test_data = self.__reshape_ditigs_matrix(test_loader.test_data.numpy(), 500)
        self.debug_test_labels = self.__reshape_labels(test_loader.test_labels.numpy(), 500)

    def __reshape_ditigs_matrix(self, X, size=None):
        """
        :param X: the matrix to be reshaped each of shape(28x28)
        :return: X_train reshaped with size (784x1)
        """
        if size:
            reduced_X = X[:size]
            reduced_X = np.array([np.reshape(x/255, (784, 1)) for x in reduced_X])
            return reduced_X
        X = np.array([np.reshape(x / 255, (784, 1)) for x in X])
        return X

    def __reshape_labels(self, labels, size=None):
        if size:
            reduced_labels = np.array(labels[:size]).reshape((size, 1))
            y_vectorized = np.zeros((size, 10, 1))
            for digit, y_vector in zip(reduced_labels, y_vectorized):
                y_vector[digit] = 1
            return y_vectorized

        size = labels.shape[0]
        reduced_labels = np.array(labels[:size]).reshape((size, 1))
        y_vectorized = np.zeros((10, size))
        for digit, y_vector in zip(reduced_labels, y_vectorized.transpose()):
            y_vector[digit] = 1
        return y_vectorized


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

