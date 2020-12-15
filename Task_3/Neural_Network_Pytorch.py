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
        train_labels = self.__train_loader.train_labels
        return train_data[:limit].float(), train_labels[:limit].long()

    def get_test_data(self, size):
        limit = int(10000 * size)
        test_data = th.reshape(self.__test_loader.test_data, (-1, 784))
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


# device = th.device('cuda' if th.cuda.is_available() else 'cpu')
# print(device)
# dataset = Dataset()
# X_train, y_train = dataset.get_train_data(size=1)
# X_test, y_test = dataset.get_test_data(size=1)
# custom_nn = CustomNetwork(3, [784, 256, 10])
# optimizer = optim.SGD(custom_nn.parameters(), lr=0.001)
# optimizer = optim.Adam(custom_nn.parameters(), lr=0.001, weight_decay=0.1)
# custom_nn.fit(X_train, y_train, optimizer=optimizer, epochs=10)
# _, _, _ = custom_nn.evaluate(X_test, y_test)


# Improvements: trying out different architectures
dataset = Dataset()
X_train, y_train = dataset.get_train_data(size=1)
X_test, y_test = dataset.get_test_data(size=1)

net_1 = CustomNetwork(3, [784, 256, 10], dropout=True)
net_2 = CustomNetwork(3, [784, 128, 10], dropout=True)
net_3 = CustomNetwork(4, [784, 512, 128, 10], dropout=True)
net_4 = CustomNetwork(3, [784, 256, 10], dropout=False)
net_5 = CustomNetwork(3, [784, 128, 10], dropout=False)
net_6 = CustomNetwork(4, [784, 512, 128, 10], dropout=False)
neural_networks = [net_1, net_2, net_3, net_4, net_5, net_6]

final_scores = []

for index, my_nn in enumerate(neural_networks):
    opt1 = optim.SGD(my_nn.parameters(), lr=0.001, momentum=0.9, weight_decay=0.1)
    opt2 = optim.Adam(my_nn.parameters(), lr=0.001, weight_decay=0.01)
    opt3 = optim.Adagrad(my_nn.parameters(), lr=0.01, weight_decay=0.01)
    optimizers = [('SGD', opt1), ('ADAM', opt2), ('ADAGRAD', opt3)]
    for my_opt in optimizers:
        print(f'Training the {index+1} Neural Network using {my_opt[0]} as optimizer')
        start_time = time.process_time()
        my_nn.fit(X_train, y_train, optimizer=my_opt[1], epochs=10)
        correct, errors, accuracy, cm = my_nn.evaluate(X_test, y_test)
        execution_time = time.process_time() - start_time
        final_scores.append(('Network'+str(index), execution_time, cm, correct, errors, accuracy))
        print('{index} Neural Network trained in {execution_time} ms'.format(index=index+1,
                                                                                  execution_time=execution_time))




