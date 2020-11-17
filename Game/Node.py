import sys

class Node:

    def __init__(self, time, location):
        self.__time = time
        self.__visited = False
        self.__neighbours = []
        self.__location = location
        self.__distance = 0
        self.__f_score = sys.maxsize
        self.__predecessor = None

    @property
    def predecessor(self):
        return self.__predecessor

    @property
    def f_score(self):
        return self.__f_score

    @property
    def distance(self):
        return self.__distance

    @property
    def location(self):
        return self.__location

    @property
    def time(self):
        return self.__time

    @property
    def visited(self):
        return self.__visited

    @property
    def neighbours(self):
        return self.__neighbours

    @predecessor.setter
    def predecessor(self, new_predecessor):
        self.__predecessor = new_predecessor

    @f_score.setter
    def f_score(self, new_f_score):
        self.__f_score = new_f_score

    @distance.setter
    def distance(self, new_distance):
        self.__distance = new_distance

    @visited.setter
    def visited(self, new_value):
        self.__visited = new_value

    @time.setter
    def time(self, new_time):
        self.__time = new_time

    def add_neighbor(self, new_neighbor):
        self.neighbours.append(new_neighbor)