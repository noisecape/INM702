from Task_2 import Neural_Network as my_nn

# import pytorch module
import torch as th
# import neural network from pytorch
import torch.nn as nn
# import non linear activation function
import torch.nn.functional as F
# import optimizer for training
import torch.optim as optim
from sklearn.metrics import accuracy_score


class CustomNetwork(nn.Module):

    def __init__(self, n_layers, n_hidden_nodes):
        super(CustomNetwork, self).__init__()
        self.n_layers = n_layers
        self.n_hidden_nodes = n_hidden_nodes
        self.act_functions = self.__create_layers(self.n_hidden_nodes)

    def __create_layers(self, n_hidden_nodes):
        act_functions = nn.ModuleList()
        for i in range(len(n_hidden_nodes)-1):
            a_func = nn.Linear(n_hidden_nodes[i], n_hidden_nodes[i+1])
            act_functions.append(a_func)
        return act_functions

    def __forward(self, x):
        output = x
        for i, a in enumerate(self.act_functions):
            if i == len(self.act_functions) - 1:
                break
            else:
                output = F.relu(a(output))
        return output

    def fit(self, X, y, optimizer, loss_f=nn.CrossEntropyLoss(), epochs=100):
        self.train()
        for e in range(epochs):
            optimizer.zero_grad()
            loss_history = []
            output = self.__forward(X)
            loss = loss_f(output, y)
            loss_history.append(loss)
            loss.backward()
            optimizer.step()
            print(loss)

    def evaluate(self, X_test, y_test):
        self.eval()
        y_pred = self.__forward(X_test)
        score = self.eval()
        print(score)




dataset = my_nn.Dataset(False)
X_train = th.from_numpy(dataset.debug_train_data.transpose()).float()
y_train = th.from_numpy(dataset.debug_train_labels)
X_test = th.from_numpy(dataset.debug_test_data.transpose()).float()
y_test = th.from_numpy(dataset.debug_test_labels)
custom_nn = CustomNetwork(4, [784, 512, 256, 10])
optimizer = optim.SGD(custom_nn.parameters(), lr=0.1)
custom_nn.fit(X_train, y_train, optimizer=optimizer, epochs=10)
custom_nn.evaluate(X_test, y_test)
