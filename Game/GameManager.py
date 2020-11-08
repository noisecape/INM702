import enum
from Game import Agent
from Game import Board


class GoalLocation(enum.Enum):
    GOAL_LOCATION = (0, 0)


class GameManager:
    time = 0
    game_over = False

    def __init__(self, agent, board):
        self.__agent = agent
        self.__board = board

    def start_game(self):
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

    def check_gameover(self):
        """
        This function checks whether the player has reached the bottom-right corner.
        If this is the case, then is game over.

        :return:
        True if is gameover, otherwise False

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
