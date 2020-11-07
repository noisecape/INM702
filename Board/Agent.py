from enum import IntEnum
import Board as bd

class AgentProperty(IntEnum):
    START_X = 0
    START_Y = 0


class PlayerStrategy(IntEnum):
    NAIVE = 0
    DIJKSTRA = 1
    ANT_COLONY = 2


class Agent:

    def __init__(self):
        self.__x = AgentProperty.START_X.value
        self.__y = AgentProperty.START_Y.value
        self.__strategy = PlayerStrategy.NAIVE
        self.__grid_status = bd.Board().grid

    @property
    def grid_status(self):
        return self.__grid_status

    @property
    def strategy(self):
        return self.__strategy

    @property
    def x(self):
        return self.__x

    @property
    def y(self):
        return self.__y

    @strategy.setter
    def strategy(self, new_strategy):
        self.__strategy = new_strategy

    @x.setter
    def x(self, new_x):
        self.__x = new_x

    @y.setter
    def y(self, new_y):
        self.__y = new_y

    def apply_strategy(self):
        print("...computing strategy...")



