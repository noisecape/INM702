from Game.Utilities import PlayerStrategy, Moves
import sys
import math
import numpy as np


class Agent:

    def __init__(self):
        self.__current_location = (0, 0)
        self.__strategy = PlayerStrategy.NAIVE.name
        self.__previous_location = (0, 0)

    def get_time_value(self):
        x = self.current_location[0]
        y = self.current_location[1]
        return self.grid[x][y]

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

    def get_possible_moves(self, board):
        """
        This function checks for all the possible moves that the agent at a given time can do.
        The agent cannot come back to the previous location nor exceed the bounds of the board.
        :return: possible_moves: list of all the possible moves
        """

        current_x = self.current_location[0]
        current_y = self.current_location[1]
        possible_moves = {}

        if current_x - 1 >= 0 and not self.is_previous(current_x - 1, current_y):
            possible_moves.update({Moves.UP.name: (current_x - 1, current_y)})
        if current_x + 1 <= len(board) - 1 and not self.is_previous(current_x + 1, current_y):
            possible_moves.update({Moves.DOWN.name: (current_x + 1, current_y)})
        if current_y - 1 >= 0 and not self.is_previous(current_x, current_y - 1):
            possible_moves.update({Moves.LEFT.name: (current_x, current_y - 1)})
        if current_y + 1 <= len(board[0]) - 1 and not self.is_previous(current_x, current_y + 1):
            possible_moves.update({Moves.RIGHT.name: (current_x, current_y + 1)})
        return possible_moves

    def distance_from_goal(self, moves, board):
        """
        Compute the euclidean distance from all the future's move location to the goal.
        :return:
        distances: a dictionary storing all the distances according to the possible move.
        """

        distances = {}

        goal_x = len(board) - 1
        goal_y = len(board[0]) - 1

        surrounding_area = [(goal_x, goal_y-1), (goal_x-1, goal_y)]

        for key, value in moves.items():
            if key == Moves.UP.name:
                up_x = value[0]
                up_y = value[1]
                if (up_x-1, up_y) in surrounding_area:
                    distance = self.get_euclidean_distance(goal_x, up_x, goal_y, up_y)
                else:
                    distance = self.get_euclidean_distance(goal_x, up_x, goal_y, up_y) + (np.random.uniform(0.1, 1) * board[up_x][up_y])
                distances.update({Moves.UP.name: distance})
            elif key == Moves.DOWN.name:
                down_x = value[0]
                down_y = value[1]
                if (down_x+1, down_y) in surrounding_area:
                    distance = self.get_euclidean_distance(goal_x, down_x, goal_y, down_y)
                else:
                    distance = self.get_euclidean_distance(goal_x, down_x, goal_y, down_y) + (np.random.uniform(0.1, 1) * board[down_x][down_y])
                distances.update({Moves.DOWN.name: distance})
            elif key == Moves.LEFT.name:
                left_x = value[0]
                left_y = value[1]
                if (left_x, left_y-1) in surrounding_area:
                    distance = self.get_euclidean_distance(goal_x, left_x, goal_y, left_y)
                else:
                    distance = self.get_euclidean_distance(goal_x, left_x, goal_y, left_y) + (np.random.uniform(0.1, 1) * board[left_x][left_y])
                distances.update({Moves.LEFT.name: distance})
            else:
                right_x = value[0]
                right_y = value[1]
                if (right_x, right_y+1) in surrounding_area:
                    distance = self.get_euclidean_distance(goal_x, right_x, goal_y, right_y)
                else:
                    distance = self.get_euclidean_distance(goal_x, right_x, goal_y, right_y) + (np.random.uniform(0.1, 1) * board[right_x][right_y])
                distances.update({Moves.RIGHT.name: distance})
        return distances

    def get_euclidean_distance(self, x1, x2, y1, y2):
        """
        Compute the Euclidean distance
        :param x1: x coordinate for the agent
        :param x2: x coordinate for the goal
        :param y1: y coordinate for the agent
        :param y2: y coordinate for the goal
        :return: the value of the euclidean distance
        """
        eu_distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        return eu_distance

    def apply_naive(self, board):
        pattern = [self.current_location]
        while self.current_location != (len(board) - 1, len(board[0]) - 1):
            possible_moves = self.get_possible_moves(board)
            distances = self.distance_from_goal(possible_moves, board)
            smallest_distance = min(distances.values())
            possible_best_moves = [move for move in distances if distances[move] == smallest_distance]
            best_move = possible_best_moves[0]
            next_location = possible_moves[best_move]
            self.__previous_location = self.__current_location
            self.__current_location = next_location
            pattern.append(next_location)
        return pattern

    def __get_closest_node(self, unseen_nodes):
        closest_node = None
        for node in unseen_nodes:
            if closest_node is None:
                closest_node = node
            elif node.distance < closest_node.distance:
                closest_node = node
        return closest_node

    def apply_dijkstra(self, graph):
        unseen_nodes = []
        start = graph[0]
        destination = graph[-1]
        for node in graph:
            if node == start:
                node.distance = 0
            else:
                node.distance = sys.maxsize
        unseen_nodes = graph[:]

        while len(unseen_nodes) != 0:
            shortest_node = self.__get_closest_node(unseen_nodes)
            for neighbor in shortest_node.neighbours:
                new_distance = shortest_node.distance + neighbor.time
                if neighbor.distance > new_distance:
                    neighbor.distance = new_distance
                    neighbor.predecessor = shortest_node
            unseen_nodes.remove(shortest_node)

        return self.get_shortest_path(destination, start)

    def apply_a_star(self, graph):
        start_node = graph[0]
        dest_node = graph[-1]
        priority_queue = [start_node]
        g_score = {}
        for node in graph:
            if node == start_node:
                g_score[node] = 0
                node.f_score = self.get_euclidean_distance(start_node.location[0], dest_node.location[0], start_node.location[1], dest_node.location[1])
            else:
                g_score[node] = sys.maxsize

        while priority_queue:
            current = priority_queue.pop(-1)
            if current == dest_node:
                "destination reached"
                break
            for neighbor in current.neighbours:
                new_g_score = g_score[current] + neighbor.time
                if new_g_score < g_score[neighbor]:
                    neighbor.predecessor = current
                    g_score[neighbor] = new_g_score
                    neighbor.f_score = g_score[neighbor] + self.get_euclidean_distance(neighbor.location[0], dest_node.location[0], neighbor.location[1], dest_node.location[1])
                    if neighbor not in priority_queue:
                        priority_queue.append(neighbor)
                        priority_queue.sort(key=lambda x: x.f_score, reverse=True)
        return self.get_shortest_path(dest_node, start_node)

    def get_shortest_path(self, dest_node, start_node):
        path = []
        current_node = dest_node
        while current_node != start_node:
            path.append(current_node.location)
            current_node = current_node.predecessor
        path.reverse()
        path.insert(0, (0, 0))
        return path

    def apply_strategy(self, graph, board):
        """
        This function computes the next move for the agent according to the choosen strategy.
        :return:
        """
        if self.strategy == PlayerStrategy.NAIVE.name:
            return self.apply_naive(board)
        elif self.strategy == PlayerStrategy.DIJKSTRA.name:
            start = graph[0]
            destination = graph[-1]
            return self.apply_dijkstra(graph)

        else:
            return self.apply_a_star(graph)
