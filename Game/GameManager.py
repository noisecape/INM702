from Game.Utilities import PlayerStrategy
from Game import Agent, Node
import numpy as np
import time as tm

class GameManager:
    time = 0
    game_over = False

    def __init__(self, n_rows, n_columns):
        self.__agent = None
        self.__n_rows = n_rows
        self.__n_columns = n_columns
        self.__board = np.random.randint(0, 10, (self.__n_rows, self.__n_columns))
        self.__graph = self.create_graph()

    @property
    def agent(self):
        return self.__agent

    @agent.setter
    def agent(self, new_agent):
        self.__agent = new_agent

    def create_graph(self):
        list_of_nodes = []
        for i in range(self.__n_rows):
            for j in range(self.__n_columns):
                node = Node.Node(self.__board[i][j], (i, j))
                list_of_nodes.append(node)
        self.__find_neighbors(list_of_nodes)
        return list_of_nodes

    def __find_neighbors(self, list_of_nodes):
        """
        This function iterate through the adjacency list and find all the neighbours for each node.
        :return:
        void
        """
        for node in list_of_nodes:
            x_pos = node.location[0]
            y_pos = node.location[1]
            if x_pos - 1 >= 0:
                # find the node in the list of nodes
                # add it as a neighbor of the current node
                neighbor = self.__find_neighbor_at(x_pos - 1, y_pos, list_of_nodes)
                node.add_neighbor(neighbor)
            if x_pos + 1 <= self.__n_rows - 1:
                neighbor = self.__find_neighbor_at(x_pos + 1, y_pos, list_of_nodes)
                node.add_neighbor(neighbor)
            if y_pos - 1 >= 0:
                neighbor = self.__find_neighbor_at(x_pos, y_pos - 1, list_of_nodes)
                node.add_neighbor(neighbor)
            if y_pos + 1 <= self.__n_columns - 1:
                neighbor = self.__find_neighbor_at(x_pos, y_pos + 1, list_of_nodes)
                node.add_neighbor(neighbor)

    def __find_neighbor_at(self, x, y, list_of_nodes):
        """
        Find the neighbor at a specific location.
        :param x: the x coordinate of the neighbor to be found
        :param y: the y coordinate of the neighbor to be found
        :return:
        node: the neighbor of a current node with the exact location (x,y)
        """
        for node in list_of_nodes:
            if node.location == (x, y):
                return node

    def start_game(self):
        """
        This function represent the main game loop. The agent before entering the main loop, applies the winning
        strategy to obtain the winning pattern, which is list of locations (x,y). At each iteration of the main
        loop, the optimal move is retrieved from the winning path. To respect the rules of the game, an internal
        loop is created: at each iteration the duration of the game is increased by one instance of time. The loop
        ends when the time spent in that location is equal to the amount of time required to wait in that location.
        Finally, the agent location is updated with the optimal move retrieved in the winning_path.
        :return: void
        """
        GameManager.time = 0
        print("Game started")
        time_spent = 0
        start_time = tm.process_time()
        winning_path = self.__agent.apply_strategy(self.__graph, self.__board)
        execution_time = tm.process_time() - start_time
        for location in winning_path:
            x = location[0]
            y = location[1]
            time_to_wait = self.__board[x][y]
            while time_spent != time_to_wait:
                GameManager.time += 1
                time_spent += 1
            self.__agent.current_location = location
            GameManager.time += 1
            time_spent = 0
        self.__print_grid(winning_path)
        print(f"Total time required: {GameManager.time}")
        return GameManager.time, execution_time

    def __print_horizontal_border(self):
        for i in range(self.__n_columns):
            print('+---', end='')

    def __print_grid(self, pattern):
        for i, row in enumerate(self.__board):
            self.__print_horizontal_border()
            print(end='+')
            print()
            print(end='|')
            for j, column in enumerate(row):
                if (i, j) in pattern:
                    G = '\033[32m'
                    W = '\033[0m'
                    print(' ' + G + 'X' + W + ' ', end='|')
                else:
                    print(' ' + str(column) + ' ', end='|')
            print()
        self.__print_horizontal_border()
        print(end='+')
        print()



naive_trials = []
dijkstra_trials = []
a_star_trials = []
trials = [naive_trials, dijkstra_trials, a_star_trials]
i = 0
for e in range(70):
    game_manager = GameManager(16+i, 8+i)
    print('NAIVE')
    player2 = Agent.Agent()
    player2.strategy = PlayerStrategy.NAIVE.name
    game_manager.agent = player2
    path_time, execution_time = game_manager.start_game()
    trials[0].append((path_time, execution_time))

    print('DIJKSTRA')
    player1 = Agent.Agent()
    player1.strategy = PlayerStrategy.DIJKSTRA.name
    game_manager.agent = player1
    path_time, execution_time = game_manager.start_game()
    trials[1].append((path_time, execution_time))

    print('A*')
    player3 = Agent.Agent()
    player3.strategy = PlayerStrategy.A_STAR.name
    game_manager.agent = player3
    path_time, execution_time = game_manager.start_game()
    trials[2].append((path_time, execution_time))
    i += 1

print(f'## NAIVE APPROACH: Average time to find the path {np.mean(trials[0][0])}')
print(f'## DIJKSTRA APPROACH: Average time to find the path {np.mean(trials[1][0])}')
print(f'## A* APPROACH: Average time to find the path {np.mean(trials[2][0])}')
print()
print(f'## NAIVE APPROACH: Average time to running algorithm {np.mean(trials[0][1])}')
print(f'## DIJKSTRA APPROACH: Average time to running algorithm {np.mean(trials[1][1])}')
print(f'## A* APPROACH: Average time to running algorithm {np.mean(trials[2][1])}')

