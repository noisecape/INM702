from Game.Utilities import BoardProperties
from Game import Node
import numpy as np


class Board:

    def __init__(self):
        self.__n_rows = BoardProperties.N_ROWS.value
        self.__n_columns = BoardProperties.N_COLUMNS.value
        self.__grid = np.random.randint(0, 10, (self.__n_rows, self.__n_columns))
        self.__list_of_nodes = []

    def create_graph(self):
        """
        This function generates n*n instances of the class Node and store them in a list. By doing so,
        it is possible to represent the matrix as a graph using the adjacency list implementation.
        The graph representation is required for the implementation of the Dijkstra algorithm.
        :return:
        list_of_nodes: the adjacency list which is responsible for the representation of the graph
        """
        for i in range(BoardProperties.N_ROWS.value):
            for j in range(BoardProperties.N_COLUMNS.value):
                node = Node.Node(self.grid[i][j], (i, j))
                self.list_of_nodes.append(node)
        self.__find_neighbors()
        return self.list_of_nodes

    def __find_neighbors(self):
        """
        This function iterate through the adjacency list and find all the neighbours for each node.
        :return:
        void
        """
        for node in self.list_of_nodes:
            x_pos = node.location[0]
            y_pos = node.location[1]
            if x_pos - 1 >= 0:
                # find the node in the list of nodes
                # add it as a neighbor of the current node
                neighbor = self.__find_neighbor_at(x_pos - 1, y_pos)
                node.add_neighbor(neighbor)
            if x_pos + 1 <= BoardProperties.N_ROWS.value:
                neighbor = self.__find_neighbor_at(x_pos + 1, y_pos)
                node.add_neighbor(neighbor)
            if y_pos - 1 >= 0:
                neighbor = self.__find_neighbor_at(x_pos, y_pos - 1)
                node.add_neighbor(neighbor)
            if y_pos + 1 <= BoardProperties.N_COLUMNS.value:
                neighbor = self.__find_neighbor_at(x_pos, y_pos + 1)
                node.add_neighbor(neighbor)

    def __find_neighbor_at(self, x, y):
        """
        Find the neighbor at a specific location.
        :param x: the x coordinate of the neighbor to be found
        :param y: the y coordinate of the neighbor to be found
        :return:
        node: the neighbor of a current node with the exact location (x,y)
        """
        for node in self.list_of_nodes:
            if node.location == (x, y):
                return node

    @property
    def list_of_nodes(self):
        return self.__list_of_nodes

    @property
    def grid(self):
        return self.__grid

    @property
    def n_rows(self):
        return self.__n_rows

    @property
    def n_columns(self):
        return self.__n_columns

    @grid.setter
    def grid(self, new_grid):
        self.__grid = new_grid

    @n_rows.setter
    def n_rows(self, new_rows):
        self.__n_rows = new_rows

    @n_columns.setter
    def n_columns(self, new_columns):
        self.__n_columns = new_columns

    def __print_horizontal_border(self):
        for i in range(self.__n_columns):
            print('+---', end='')

    def print_grid(self, pattern):
        for i, row in enumerate(self.__grid):
            self.__print_horizontal_border()
            print(end='+')
            print()
            print(end='|')
            for j, column in enumerate(row):
                if (i, j) in pattern:
                    G = '\033[32m'
                    W = '\033[0m'
                    print(' ' + G + 'X' + W + ' ', end='|')
                else:
                    print(' '+str(column)+' ', end='|')
            print()
        self.__print_horizontal_border()
        print(end='+')
        print()

    def get_element(self, row, column):
        return self.__grid[row][column]


