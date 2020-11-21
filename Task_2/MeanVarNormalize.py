class MeanVarNormalize():

    def __init__(self, X):
        self.X = X

    def compute_mean(self):
        temp = 0
        for row in self.X:
            for column in row:
                temp += column
        return temp / len(self.X)

    def compute_std(self, mean):
        var = 0
        for row in self.X:
            for column in row:
                var += (column - mean) ** 2

        return var ** (1 / 2)

    def standardization(self):
        mean = self.compute_mean()
        std = self.compute_std(mean)
        Z = (self.X[:, :] - mean) / std
        return Z