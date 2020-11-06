import numpy as np

class Board:

    def __init__(self, width, height):
        self.__width = width
        self.__height = height
        self.__board = self.init_board()

    def init_board(self):
        board = [[]]
        for row in self.__width:
            for column in self.__column:
                board[row][column] = np.random.randint(0,10)
        return board


    @property
    def get_board(self):
        return self.__board

    def print_board(self):



board = Board()
board.prin