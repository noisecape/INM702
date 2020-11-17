from Game.Utilities import PlayerStrategy
from Game import Agent, Board


class GameManager:
    time = 0
    game_over = False

    def __init__(self, agent, board):
        self.__agent = agent
        self.__board = board

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
        print("Game started")
        time_spent = 0
        winning_path = self.__agent.apply_strategy(self)
        for location in winning_path:
            x = location[0]
            y = location[1]
            time_to_wait = self.__agent.grid[x][y]
            while time_spent != time_to_wait:
                GameManager.time += 1
                time_spent += 1
            self.__agent.current_location = location
            GameManager.time += 1
            time_spent = 0
        self.__board.print_grid(winning_path)
        print(f"Total time required: {GameManager.time}")

    def get_graph_representation(self):
        return self.__board.create_graph()


board_instance = Board.Board(11, 15)
grid = board_instance.grid
player1 = Agent.Agent(grid)
player1.strategy = PlayerStrategy.DIJKSTRA.name
game_manager = GameManager(player1, board_instance)
game_manager.start_game()

player2 = Agent.Agent(grid)
player2.strategy = PlayerStrategy.NAIVE.name
game_manager = GameManager(player2, board_instance)
game_manager.start_game()

player3 = Agent.Agent(grid)
player3.strategy = PlayerStrategy.A_STAR.name
game_manager = GameManager(player3, board_instance)
game_manager.start_game()
