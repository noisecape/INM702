import enum
from Game import Board


class AgentProperty(enum.Enum):
    START_LOCATION = (0, 0)


class PlayerStrategy(enum.IntEnum):
    NAIVE = 0
    DIJKSTRA = 1
    ANT_COLONY = 2


class Agent:

    def __init__(self, grid):
        self.__current_location = AgentProperty.START_LOCATION.value
        self.__strategy = PlayerStrategy.NAIVE.value
        self.__grid = grid
        self.__previous_location = (0, 0)

    def get_time_value(self):
        x = self.current_location[0]
        y = self.current_location[1]
        return self.grid[x][y]

    @property
    def grid(self):
        return self.__grid

    @property
    def strategy(self):
        return self.__strategy

    @property
    def current_location(self):
        return self.__current_location

    @strategy.setter
    def strategy(self, new_strategy):
        self.__strategy = new_strategy

    @current_location.setter
    def x(self, new_location):
        self.__current_location = new_location

    def apply_strategy(self):
        print("...computing strategy...")
        print("create a dictionary of the possible moves") # {'U':(x,y), 'D':(x,y), ...}
        print("compute the euclidean distance for all the possible moves and store them in a dictionary")
        # {'U': eu_dist_up, 'D': eu_dist_down, ...}
        print("choose the smallest value from the euclidean distances")
        print("if there are some equal distances, "
              "choose the one with the smallest amount of time to spend on the location.")
        print("if time are equals, choose the first one.")



