from Game.Utilities import GoalLocation, AgentProperty, PlayerStrategy, Borders, Moves


class Agent:

    def __init__(self, grid):
        self.__current_location = AgentProperty.START_LOCATION.value
        self.__strategy = PlayerStrategy.NAIVE.value
        self.__grid = grid
        self.__previous_location = (0, 0)
        self.__pattern = [self.__current_location]
        self.__moves = []

    def get_time_value(self):
        x = self.current_location[0]
        y = self.current_location[1]
        return self.grid[x][y]

    @property
    def pattern(self):
        return self.__pattern

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
    def current_location(self, new_location):
        self.__current_location = new_location

    def is_previous(self, x, y):
        """
        Chek if the location provided in the parameters is the previous location
        :param x: the x coordinates
        :param y: the y coordinates
        :return: True if the location (x,y) is the previous location
        """
        potential_previous = (x, y)
        if self.__current_location == (0, 0):
            return False
        if potential_previous == self.__previous_location:
            return True
        else:
            return False

    def get_possible_moves(self):
        # STILL HAVE TO CHECK THE PREVIOUS LOCATION
        current_x = self.current_location[0]
        current_y = self.current_location[1]
        possible_moves = {}
        # check can move up
        if current_x - 1 >= Borders.ROW_LOWER_LIMIT.value and not self.is_previous(current_x - 1, current_y):
            possible_moves.update({Moves.UP.name: (current_x - 1, current_y)})
        if current_x + 1 <= Borders.ROW_UPPER_LIMIT.value and not self.is_previous(current_x + 1, current_y):
            possible_moves.update({Moves.DOWN.name: (current_x + 1, current_y)})
        if current_y - 1 >= Borders.COLUMN_LOWER_LIMIT.value and not self.is_previous(current_x, current_y - 1):
            possible_moves.update({Moves.LEFT.name: (current_x, current_y - 1)})
        if current_y + 1 <= Borders.COLUMN_UPPER_LIMIT.value and not self.is_previous(current_x, current_y + 1):
            possible_moves.update({Moves.RIGHT.name: (current_x, current_y + 1)})
        return possible_moves

    def distance_from_goal(self, moves):
        """
        Compute the euclidean distance from all the future's move location to the goal.
        :return:
        distances: a dictionary storing all the distances according to the possible move.
        """

        distances = {}

        goal_x = GoalLocation.GOAL_LOCATION.value[0]
        goal_y = GoalLocation.GOAL_LOCATION.value[1]

        for key, value in moves.items():
            if key == Moves.UP.name:
                up_x = value[0]
                up_y = value[1]
                distance = self.get_euclidean_distance(goal_x, up_x, goal_y, up_y)
                distances.update({Moves.UP.name: distance})
            elif key == Moves.DOWN.name:
                down_x = value[0]
                down_y = value[1]
                distance = self.get_euclidean_distance(goal_x, down_x, goal_y, down_y)
                distances.update({Moves.DOWN.name: distance})
            elif key == Moves.LEFT.name:
                left_x = value[0]
                left_y = value[1]
                distance = self.get_euclidean_distance(goal_x, left_x, goal_y, left_y)
                distances.update({Moves.LEFT.name: distance})
            else:
                right_x = value[0]
                right_y = value[1]
                distance = self.get_euclidean_distance(goal_x, right_x, goal_y, right_y)
                distances.update({Moves.RIGHT.name: distance})

        return distances

    def get_euclidean_distance(self, x1, x2, y1, y2):
        eu_distance = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 1 / 2
        return eu_distance

    def apply_strategy(self):
        possible_moves = self.get_possible_moves()
        distances = self.distance_from_goal(possible_moves)
        smallest_distance = min(distances.values())
        possible_best_moves = [move for move in distances if distances[move] == smallest_distance]
        best_move = possible_best_moves[0]
        print(f"The best options are: {best_move}")
        next_location = possible_moves[best_move]
        print("if there are some equal distances, "
              "choose the one with the smallest amount of time to spend on the location.")
        self.__previous_location = self.__current_location
        self.__current_location = next_location
        self.__moves.append(best_move)
        self.__pattern.append(next_location)
