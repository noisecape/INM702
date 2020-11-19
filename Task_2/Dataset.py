from torchvision.datasets import MNIST


class Dataset:
    """
    The class that wraps the MNIST dataset downloaded and loaded by pytorch-vision.
    Each digit is represented as a vector of 28x28 pixels, where each pixel can have values between 0-255.
    In total there are 60000 elements in the training set while the elements in the test set are 10000.
    """
    def __init__(self):
        train_loader = MNIST(root='.', download=True)
        test_loader = MNIST(root='.', train=False)
        self.__train_data = train_loader.train_data
        self.__train_labels = train_loader.train_labels
        self.__test_data = test_loader.test_data
        self.__test_labels = test_loader.test_labels

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

