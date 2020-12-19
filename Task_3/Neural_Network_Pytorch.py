from Task_2 import Neural_Network as my_nn
import numpy as np
# import pytorch module
import torch as th
# import neural network from pytorch
import torch.nn as nn
# import non linear activation function
import torch.nn.functional as F
# import optimizer for training
import torch.optim as optim
from torchvision.datasets import MNIST
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import time


class Dataset(MNIST):

    def __init__(self):
        self.__train_loader = MNIST(root='.', download=True)
        self.__test_loader = MNIST(root='.', train=False)

    def get_train_data(self, size):
        limit = int(60000*size)
        train_data = th.reshape(self.__train_loader.train_data, (-1, 784))
        for x in train_data:
            x = x/255
        train_labels = self.__train_loader.train_labels
        return train_data[:limit].float(), train_labels[:limit].long()

    def get_test_data(self, size):
        limit = int(10000 * size)
        test_data = th.reshape(self.__test_loader.test_data, (-1, 784))
        for x in test_data:
            x = x/255
        test_labels = self.__test_loader.test_labels
        return test_data[:limit].float(), test_labels[:limit].long()


class CustomNetwork(nn.Module):

    def __init__(self, n_layers, n_hidden_nodes, dropout=False):
        super(CustomNetwork, self).__init__()
        self.n_layers = n_layers
        self.n_hidden_nodes = n_hidden_nodes
        self.dropout = dropout
        self.act_functions = self.__create_layers(self.n_hidden_nodes)

    def __create_layers(self, n_hidden_nodes):
        # since we are dealing with gray scale images, the input of the first convolutional layer is 1.
        act_functions = nn.ModuleList()
        if self.dropout:
            drop_layer1 = nn.Dropout(0.1)
            act_functions.append(drop_layer1)

        for i in range(len(n_hidden_nodes)-1):
            if i == len(n_hidden_nodes)-1:
                if self.dropout:
                    drop_layer2 = nn.Dropout(0.2)
                    act_functions.append(drop_layer2)
            a_func = nn.Linear(n_hidden_nodes[i], n_hidden_nodes[i+1])
            act_functions.append(a_func)
        return act_functions

    def __forward(self, x):
        output = x
        for i, a in enumerate(self.act_functions):
            if i == len(self.act_functions) - 1:
                output = a(output)
                break
            else:
                output = F.relu(a(output))
        return output

    def fit(self, X, y, optimizer, loss_f=nn.CrossEntropyLoss(), epochs=100, batch_size=100):
        self.train()
        N = X.shape[0]
        epoch_history = []
        for e in range(epochs):
            # implement mini batch
            X_random, y_random = self.randomize(X, y)
            X_random = th.split(X_random, batch_size)
            y_random = th.split(y_random, batch_size)
            batch_history = []
            for x_batch, y_batch in zip(X_random, y_random):
                optimizer.zero_grad()
                output = self.__forward(x_batch)
                loss = loss_f(output, y_batch)
                loss.backward()
                optimizer.step()
                batch_history.append(loss.item())
            avg_loss = np.sum(batch_history) / N
            epoch_history.append(avg_loss)
            print(avg_loss)

    def randomize(self, X, y):
        X = X.numpy()
        y = y.numpy()
        perm_indices = np.random.permutation(X.shape[0])
        X_shuffled = X[perm_indices, :]
        y_shuffled = y[perm_indices]
        return th.from_numpy(X_shuffled).float(), th.from_numpy(y_shuffled).long()

    def evaluate(self, X_test, y_test):
        self.eval()
        output = self.__forward(X_test)
        y_pred = th.argmax(output, dim=1)
        correct = 0
        for pred, true in zip(y_pred, y_test):
            correct += 1 if pred == true else 0
        errors = y_test.shape[0]-correct
        accuracy = correct/y_test.shape[0]
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        self.print_confusion_matrix(cm)
        print(f'Correctly classified: {correct}')
        print(f'Misclassified: {errors}')
        print(f'Accuracy: {accuracy}')
        return correct, errors, accuracy, cm

    def print_confusion_matrix(self, cm):
        display = ConfusionMatrixDisplay(cm, display_labels=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
        plt.title = 'Naive with balanced Dataset'
        display.plot(xticks_rotation=90)
        plt.ylabel = 'True Values'
        plt.xlabel = 'Predicted Values'


data = Dataset()
data.get_train_data(0.4)

