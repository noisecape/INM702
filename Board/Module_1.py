import enum
import numpy as np


class BoardProperties(enum.IntEnum):
    N_ROWS = 11
    N_COLUMNS = 15


class Board:

    def __init__(self):
        self.__n_rows = BoardProperties.N_ROWS.value
        self.__n_columns = BoardProperties.N_COLUMNS.value
        self.__grid = np.random.randint(0, 10, (self.__n_rows, self.__n_columns))

    @property
    def get_grid(self):
        return self.__grid

    @property
    def get_n_rows(self):
        return self.__n_rows

    @property
    def get_n_columns(self):
        return self.__n_columns

    def print_horizontal_border(self):
        for i in range(self.__n_columns):
            print('+---', end='')

    def print_grid(self):
        for row in self.__grid:
            self.print_horizontal_border()
            print(end='+')
            print()
            print(end='|')
            for column in row:
                print(' '+str(column)+' ', end='|')
            print()
        self.print_horizontal_border()
        print(end='+')
        print()

board = Board()
board.print_grid()