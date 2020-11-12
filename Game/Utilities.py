import enum


class Moves(enum.IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class PlayerStrategy(enum.IntEnum):
    NAIVE = 0
    DIJKSTRA = 1
    ANT_COLONY = 2
