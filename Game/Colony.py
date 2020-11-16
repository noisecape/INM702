from Game import Ant


class Colony:

    def __init__(self, n_ants, rho, alpha, beta):
        self.n_ants = n_ants
        self.rho = rho
        self.alpha = alpha
        self.beta = beta

    def get_ants(self):
        ants = []
        for _ in range(self.n_ants):
            ant = Ant.Ant()
            ants.append(ant)
        return ants
