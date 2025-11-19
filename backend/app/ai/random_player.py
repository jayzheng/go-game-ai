"""
Random AI player - makes random legal moves
Good for testing and as a baseline opponent
"""
import random
from typing import Tuple, Optional
from ..game.go_game import GoGame


class RandomPlayer:
    """AI player that makes random legal moves"""

    def __init__(self, pass_threshold: float = 0.1):
        """
        Args:
            pass_threshold: Probability of passing when board is mostly filled
        """
        self.pass_threshold = pass_threshold

    def get_move(self, game: GoGame) -> Optional[Tuple[int, int]]:
        """
        Get next move from AI
        Returns (row, col) or None to pass
        """
        legal_moves = game.get_legal_moves()

        if not legal_moves:
            return None  # Must pass

        # Occasionally pass when few moves available
        if len(legal_moves) < game.board_size and random.random() < self.pass_threshold:
            return None

        # Return random legal move
        return random.choice(legal_moves)
