class Node:

    def __init__(self, time, location):
        self.__time = time
        self.__visited = False
        self.__neighbours = []
        self.__location = location
        self.__distance = 0

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