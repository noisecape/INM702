from Task_2 import Neural_Network as my_nn

# import pytorch module
import torch as th
# import neural network from pytorch
import torch.nn as nn
# import non linear activation function
import torch.functional as F
# import optimizer for training
import torch.optim as optim

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
                output = F.softmax(a(output))
            output = F.relu(a(output))
        return output


    def fit(self, X, y, optimizer, loss=nn.CrossEntropyLoss(), epoch=100):
        output = self.__forward(X)



dataset = my_nn.Dataset()
X_train = th.from_numpy(dataset.debug_train_data)
y_train = th.from_numpy(dataset.debug_train_labels)
X_test = th.from_numpy(dataset.debug_test_data)
y_test = th.from_numpy(dataset.debug_test_labels)
custom_nn = CustomNetwork(3, [784, 128, 10])
for p in custom_nn.parameters():
    print(type(p.data), p.size())

optimizer = optim.SGD(custom_nn.parameters(), lr=0.01)
custom_nn.fit(X_train, y_train, epochs=100)