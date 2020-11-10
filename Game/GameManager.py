from Game.Utilities import GoalLocation
from Game import Agent, Board


class GameManager:
    time = 0
    game_over = False

    def __init__(self, agent, board):
        self.__agent = agent
        self.__board = board

    def start_game(self):
        """
        This function represent the main game loop. At first the manager asks the agent
        for the time value that it has to spend on a location. When the loop start, at each iteration
        the time counter is incremented. When the time counter reachs the value of the time provided
        by the agent (read from the location) then the agent is allowed to apply its strategy.
        After the agent's move, the manager checks if the condition of 'Game Over' is reached. If that's
        the case, then the loop is terminated. Otherwise the agent is asked for the next time value
        of the new location and the previous steps are repeated.
        :return: void
        """
        print("Game started")
        player_time = self.__agent.get_time_value()
        actual_time_spent = 0
        while GameManager.game_over is False:
            # Check if the time that the player has spent in the current location
            # is equal to the time that the player need to spend in that location.
            if actual_time_spent == player_time:
                print('apply strategy')
                self.__agent.apply_strategy()
                print('check if is game over')
                GameManager.game_over = self.check_gameover()
                print('reset the counter for the time to spend in the next location')
                actual_time_spent = 0
                print('get new value for the time to spend in the new location')
                player_time = self.__agent.get_time_value()
            else:
                actual_time_spent += 1

            GameManager.time += 1
        self.__board.print_grid(self.__agent.pattern)


    def check_gameover(self):
        """
        This function checks whether the player has reached the bottom-right corner.
        If this is the case, then is game over.

        :return:
        True if is game over, otherwise False

        """
        if self.__agent.current_location == GoalLocation.GOAL_LOCATION.value:
            return True
        else:
            return False


board_instance = Board.Board()
grid = board_instance.grid
player = Agent.Agent(grid)

game_manager = GameManager(player, board_instance)
game_manager.start_game()
