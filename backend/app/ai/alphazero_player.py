"""
AlphaZero AI Player - uses neural network + MCTS
"""
import numpy as np
from typing import Tuple, Optional
from ..game.go_game import GoGame
from .neural_net import GoNet
from .mcts import MCTS


class AlphaZeroPlayer:
    """AI player using AlphaGo Zero algorithm"""

    def __init__(self, model_path: Optional[str] = None, board_size: int = 9,
                 num_simulations: int = 100):
        """
        Args:
            model_path: Path to trained model checkpoint (None for random initialized model)
            board_size: Size of go board
            num_simulations: Number of MCTS simulations per move
        """
        self.board_size = board_size
        self.num_simulations = num_simulations

        # Load or create neural network
        if model_path:
            self.neural_net, _ = GoNet.load_checkpoint(model_path)
            print(f"Loaded model from {model_path}")
        else:
            self.neural_net = GoNet(board_size=board_size)
            print("Using randomly initialized model")

        self.neural_net.eval()

        # Create MCTS engine
        self.mcts = MCTS(self.neural_net, num_simulations=num_simulations)

    def get_move(self, game: GoGame, temperature: float = 0.1) -> Optional[Tuple[int, int]]:
        """
        Get next move from AI

        Args:
            game: Current game state
            temperature: Temperature for move selection (lower = more deterministic)

        Returns:
            (row, col) or None to pass
        """
        if game.game_over:
            return None

        move, _ = self.mcts.get_move_with_temp(game, temperature=temperature)
        return move

    def set_num_simulations(self, num_simulations: int):
        """Update number of MCTS simulations"""
        self.num_simulations = num_simulations
        self.mcts.num_simulations = num_simulations
