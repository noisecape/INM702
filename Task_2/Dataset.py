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

        self.debug_train_data = self.__reshape_ditigs_matrix(train_loader.train_data.numpy(), size=6000)
        self.debug_train_labels = self.__reshape_labels(train_loader.train_labels.numpy(), size=6000)
        self.debug_test_data = self.__reshape_ditigs_matrix(test_loader.test_data.numpy(), size=1000)
        self.debug_test_labels = self.__reshape_labels(test_loader.test_labels.numpy(), size=1000)

    def __reshape_ditigs_matrix(self, X, size=None):
        """
        :param X: the matrix to be reshaped each of shape(28x28)
        :return: X_train reshaped with size (784x1)
        """

        if size:
            reduced_X = np.reshape([(x/255) for x in X[:size]], (784, size))
            return reduced_X
        X = np.reshape([x/255 for x in X], (784, X.shape[0]))
        return X

    def __reshape_labels(self, labels, size=60000):
        vectorized_labels = np.zeros((10, size))
        for vector, digit in zip(vectorized_labels.transpose(), labels[:size]):
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

