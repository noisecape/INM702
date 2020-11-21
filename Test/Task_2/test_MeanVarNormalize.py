from Task_2 import MeanVarNormalize
import unittest
import numpy as np


class TestNormalizerCase(unittest.TestCase):

    def test_normalizer(self):
        X = np.random.randn(10, 10)

        normalizer = MeanVarNormalize.MeanVarNormalize(X)
        mean = normalizer.compute_mean()
        std = normalizer.compute_std(mean)
        Z = normalizer.standardization()
        self.assertAlmostEqual(np.mean(Z), 0, delta=0.09) and self.assertAlmostEqual(np.std(Z), 1, delta=0.09)

    if __name__ == '__main__':
        unittest.main()