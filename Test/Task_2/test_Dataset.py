from Task_2 import Dataset
import numpy as np
import unittest


class TestDataset(unittest.TestCase):

    def test_train_size(self):
        print(Dataset.Dataset().train_data.shape)
        true_shape = np.ndarray(shape=(60000, 28, 28))
        result = True
        for index, shape in enumerate(true_shape.shape):
            if shape != Dataset.Dataset().train_data.shape[index]:
                result = False
        assert result is True
