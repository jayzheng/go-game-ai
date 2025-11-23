"""
Go Game Logic - Implements rules for 9x9 Go board
Supports stone placement, capture, ko rule, and scoring
"""
from typing import List, Tuple, Set, Optional
from enum import Enum
import copy


class Player(Enum):
    BLACK = 1
    WHITE = 2
    EMPTY = 0


class GoGame:
    def __init__(self, board_size: int = 9, komi: float = 7.5, max_moves: Optional[int] = None):
        """
        Initialize Go game.

        Args:
            board_size: Size of the board (default 9x9)
            komi: Compensation points for White (default 7.5)
                  Standard values: 7.5 for Chinese rules, 6.5 for Japanese rules
            max_moves: Maximum number of moves before game ends (default: board_size² × 3)
                      Prevents infinite games during training
        """
        self.board_size = board_size
        self.board = [[Player.EMPTY for _ in range(board_size)] for _ in range(board_size)]
        self.current_player = Player.BLACK
        self.move_history: List[Tuple[int, int, Player]] = []
        self.captured_stones = {Player.BLACK: 0, Player.WHITE: 0}
        self.ko_point: Optional[Tuple[int, int]] = None
        self.pass_count = 0
        self.game_over = False
        self.komi = komi  # Compensation for White going second
        self.max_moves = max_moves or (board_size * board_size * 3)  # Default: 3x board size²
        self.resigned = None  # Track if a player resigned

    def is_valid_position(self, row: int, col: int) -> bool:
        """Check if position is within board bounds"""
        return 0 <= row < self.board_size and 0 <= col < self.board_size

    def get_neighbors(self, row: int, col: int) -> List[Tuple[int, int]]:
        """Get valid neighboring positions"""
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_row, new_col = row + dr, col + dc
            if self.is_valid_position(new_row, new_col):
                neighbors.append((new_row, new_col))
        return neighbors

    def get_group(self, row: int, col: int) -> Set[Tuple[int, int]]:
        """Get all connected stones of the same color (group)"""
        if self.board[row][col] == Player.EMPTY:
            return set()

        color = self.board[row][col]
        group = set()
        stack = [(row, col)]

        while stack:
            r, c = stack.pop()
            if (r, c) in group:
                continue
            group.add((r, c))

            for nr, nc in self.get_neighbors(r, c):
                if self.board[nr][nc] == color and (nr, nc) not in group:
                    stack.append((nr, nc))

        return group

    def get_liberties(self, group: Set[Tuple[int, int]]) -> int:
        """Count liberties (empty adjacent points) for a group"""
        liberties = set()
        for row, col in group:
            for nr, nc in self.get_neighbors(row, col):
                if self.board[nr][nc] == Player.EMPTY:
                    liberties.add((nr, nc))
        return len(liberties)

    def remove_captured_stones(self, opponent: Player) -> int:
        """Remove captured opponent stones and return count"""
        captured = 0
        for row in range(self.board_size):
            for col in range(self.board_size):
                if self.board[row][col] == opponent:
                    group = self.get_group(row, col)
                    if self.get_liberties(group) == 0:
                        for r, c in group:
                            self.board[r][c] = Player.EMPTY
                        captured += len(group)
        return captured

    def is_suicide_move(self, row: int, col: int, player: Player) -> bool:
        """Check if move would be suicide (no liberties and doesn't capture)"""
        # Temporarily place stone
        self.board[row][col] = player

        # Check if this move captures opponent stones
        opponent = Player.WHITE if player == Player.BLACK else Player.BLACK
        for nr, nc in self.get_neighbors(row, col):
            if self.board[nr][nc] == opponent:
                opponent_group = self.get_group(nr, nc)
                if self.get_liberties(opponent_group) == 0:
                    self.board[row][col] = Player.EMPTY
                    return False  # Captures opponent, not suicide

        # Check if placed stone has liberties
        group = self.get_group(row, col)
        has_liberties = self.get_liberties(group) > 0

        self.board[row][col] = Player.EMPTY
        return not has_liberties

    def make_move(self, row: int, col: int) -> bool:
        """
        Attempt to place a stone at the given position
        Returns True if move is valid and executed
        """
        # Check if game is over
        if self.game_over:
            return False

        # Reset pass count on actual move
        self.pass_count = 0

        # Validate position
        if not self.is_valid_position(row, col):
            return False

        if self.board[row][col] != Player.EMPTY:
            return False

        # Check ko rule
        if self.ko_point and self.ko_point == (row, col):
            return False

        # Check suicide rule
        if self.is_suicide_move(row, col, self.current_player):
            return False

        # Place stone
        self.board[row][col] = self.current_player

        # Remove captured opponent stones
        opponent = Player.WHITE if self.current_player == Player.BLACK else Player.BLACK
        captured = self.remove_captured_stones(opponent)
        self.captured_stones[self.current_player] += captured

        # Update ko point (simple ko: single stone capture)
        if captured == 1:
            # Find the captured stone position
            for r in range(self.board_size):
                for c in range(self.board_size):
                    if self.board[r][c] == Player.EMPTY:
                        neighbors = self.get_neighbors(r, c)
                        if (row, col) in neighbors:
                            # Check if this could be a ko situation
                            group = self.get_group(row, col)
                            if len(group) == 1 and self.get_liberties(group) == 1:
                                self.ko_point = (r, c)
                                break
            else:
                self.ko_point = None
        else:
            self.ko_point = None

        # Record move
        self.move_history.append((row, col, self.current_player))

        # Check if max moves reached
        if len(self.move_history) >= self.max_moves:
            self.game_over = True

        # Switch player
        self.current_player = opponent

        return True

    def pass_turn(self):
        """Pass the turn"""
        self.pass_count += 1
        self.move_history.append((-1, -1, self.current_player))  # -1, -1 indicates pass

        # Game ends after two consecutive passes
        if self.pass_count >= 2:
            self.game_over = True

        # Switch player
        self.current_player = (
            Player.WHITE if self.current_player == Player.BLACK else Player.BLACK
        )

    def resign(self, player: Player):
        """
        Player resigns the game.

        Args:
            player: The player who is resigning
        """
        self.resigned = player
        self.game_over = True

    def get_score(self) -> dict:
        """
        Calculate score using area scoring (Chinese-style rules).
        Area scoring = stones on board + surrounded empty points
        White receives komi compensation for going second.

        Returns dict with scores for both players including komi
        """
        territory = {Player.BLACK: 0, Player.WHITE: 0}
        visited = set()

        # Find territories (empty regions)
        for row in range(self.board_size):
            for col in range(self.board_size):
                if (row, col) in visited or self.board[row][col] != Player.EMPTY:
                    continue

                # Flood fill to find empty region
                region = set()
                borders = set()
                stack = [(row, col)]

                while stack:
                    r, c = stack.pop()
                    if (r, c) in region:
                        continue

                    if not self.is_valid_position(r, c):
                        continue

                    if self.board[r][c] == Player.EMPTY:
                        region.add((r, c))
                        for nr, nc in self.get_neighbors(r, c):
                            if (nr, nc) not in region:
                                stack.append((nr, nc))
                    else:
                        borders.add(self.board[r][c])

                # Assign territory if it borders only one color
                if len(borders) == 1:
                    owner = borders.pop()
                    territory[owner] += len(region)

                visited.update(region)

        # Count stones on board
        stones = {Player.BLACK: 0, Player.WHITE: 0}
        for row in range(self.board_size):
            for col in range(self.board_size):
                if self.board[row][col] != Player.EMPTY:
                    stones[self.board[row][col]] += 1

        # Area scoring: stones + territory
        # White gets komi compensation
        black_score = stones[Player.BLACK] + territory[Player.BLACK]
        white_score = stones[Player.WHITE] + territory[Player.WHITE] + self.komi

        return {
            'black': black_score,
            'white': white_score,
            'black_stones': stones[Player.BLACK],
            'white_stones': stones[Player.WHITE],
            'black_territory': territory[Player.BLACK],
            'white_territory': territory[Player.WHITE],
            'black_captures': self.captured_stones[Player.BLACK],
            'white_captures': self.captured_stones[Player.WHITE],
            'komi': self.komi
        }

    def calculate_score(self) -> Tuple[float, float]:
        """
        Calculate final scores for both players.
        Uses area scoring (Chinese rules) with komi.

        Returns:
            Tuple of (black_score, white_score)
        """
        score_dict = self.get_score()
        return score_dict['black'], score_dict['white']

    def get_legal_moves(self) -> List[Tuple[int, int]]:
        """Get all legal moves for current player"""
        legal_moves = []
        for row in range(self.board_size):
            for col in range(self.board_size):
                if self.board[row][col] == Player.EMPTY:
                    if not self.is_suicide_move(row, col, self.current_player):
                        if not (self.ko_point and self.ko_point == (row, col)):
                            legal_moves.append((row, col))
        return legal_moves

    def get_state(self) -> dict:
        """Get current game state as dictionary"""
        return {
            'board': [[cell.value for cell in row] for row in self.board],
            'current_player': self.current_player.value,
            'move_history': [(r, c, p.value) for r, c, p in self.move_history],
            'captured_stones': {
                'black': self.captured_stones[Player.BLACK],
                'white': self.captured_stones[Player.WHITE]
            },
            'game_over': self.game_over,
            'score': self.get_score() if self.game_over else None
        }

    def clone(self) -> 'GoGame':
        """Create a deep copy of the game state"""
        return copy.deepcopy(self)
