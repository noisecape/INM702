from enum import IntEnum
from Game import Agent
from Game import Board


class GoalLocation(IntEnum):
    X_LOCATION = 10
    Y_LOCATION = 14


class GameManager:
    time = 0

    def __init__(self, agent, board):
        self.__agent = agent
        self.__board = board

    def start_game(self):
        while self.__agent.x != GoalLocation.X_LOCATION and self.__agent.y != GoalLocation.Y_LOCATION:
            print("Game started")
            GameManager.time += 1
            print(GameManager.time)
            if GameManager.time is 10:
                break


grid = Board.Board()
player = Agent.Agent()
game_manager = GameManager(player, grid)
game_manager.start_game()
