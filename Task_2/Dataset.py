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

        # self.__train_data = self.__reshape_ditigs_matrix(train_loader.train_data)
        # self.__train_labels = self.__reshape_labels(train_loader.train_labels)
        # self.__test_data = self.__reshape_ditigs_matrix(test_loader.test_data)
        # self.__test_labels = self.__reshape_labels(test_loader.test_labels)

        # for debugging
        self.debug_train_data = self.__reshape_ditigs_matrix(train_loader.train_data, 2000)
        self.debug_train_labels = self.__reshape_labels(train_loader.train_labels, 2000)
        self.debug_test_data = self.__reshape_ditigs_matrix(test_loader.test_data, 500)
        self.debug_test_labels = self.__reshape_labels(test_loader.test_labels, 500)

    def __reshape_ditigs_matrix(self, X, size=None):
        """
        Reshape each element in X_train in a vector shape of (784x1)
        :param X: the matrix to be reshaped each of shape(28x28)
        :return: X_train reshaped with size (784x1)
        """
        if size:
            reduced_X = X[:size]
            reduced_X = [np.reshape(x, (784, 1)) for x in reduced_X]
            return reduced_X
        X = [np.reshape(x, (784, 1)) for x in X]
        return X

    def __reshape_labels(self, labels, size=None):
        if size:
            reduced_labels = labels[:size]
            reduced_new_labels = np.ndarray((size, 10))
            for index, label in enumerate(reduced_labels):
                digits = np.zeros(10)
                for i in range(10):
                    if i == label:
                        digits[i] = 1
                reduced_new_labels[index] = [x for x in digits]
            return reduced_new_labels

        new_labels = np.ndarray((labels.shape[0], 10))
        for index, label in enumerate(labels):
            digits = np.zeros(10)
            for i in range(10):
                if i == label:
                    digits[i] = 1
            new_labels[index] = [x for x in digits]
        return new_labels

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

