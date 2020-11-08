from Game.Utilities import BoardProperties
import numpy as np


class Board:

    def __init__(self):
        self.__n_rows = BoardProperties.N_ROWS.value
        self.__n_columns = BoardProperties.N_COLUMNS.value
        self.__grid = np.random.randint(0, 10, (self.__n_rows, self.__n_columns))

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

    def print_grid(self):
        for row in self.__grid:
            self.__print_horizontal_border()
            print(end='+')
            print()
            print(end='|')
            for column in row:
                print(' '+str(column)+' ', end='|')
            print()
        self.__print_horizontal_border()
        print(end='+')
        print()

    def get_element(self, row, column):
        return self.__grid[row][column]
