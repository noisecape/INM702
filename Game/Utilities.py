import enum


class GoalLocation(enum.Enum):
    GOAL_LOCATION = (10, 14)


class Moves(enum.IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class Borders(enum.IntEnum):
    ROW_UPPER_LIMIT = 10
    ROW_LOWER_LIMIT = 0
    COLUMN_UPPER_LIMIT = 14
    COLUMN_LOWER_LIMIT = 0


class AgentProperty(enum.Enum):
    START_LOCATION = (0, 0)


class PlayerStrategy(enum.IntEnum):
    NAIVE = 0
    DIJKSTRA = 1
    ANT_COLONY = 2


class BoardProperties(enum.IntEnum):
    N_ROWS = 11
    N_COLUMNS = 15
