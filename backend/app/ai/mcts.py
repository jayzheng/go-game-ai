"""
Monte Carlo Tree Search (MCTS) for AlphaGo Zero
Uses neural network to guide tree search
"""
import numpy as np
import math
from typing import Optional, Tuple, List
from ..game.go_game import GoGame, Player


class MCTSNode:
    """Node in the Monte Carlo search tree"""

    def __init__(self, game_state: GoGame, parent: Optional['MCTSNode'] = None,
                 prior: float = 0.0, move: Optional[Tuple[int, int]] = None):
        self.game_state = game_state.clone()
        self.parent = parent
        self.move = move  # Move that led to this node
        self.prior = prior  # Prior probability from neural network

        self.children = {}  # {move: MCTSNode}
        self.visit_count = 0
        self.value_sum = 0.0
        self.is_expanded = False

    def is_leaf(self) -> bool:
        return not self.is_expanded

    def value(self) -> float:
        """Average value of this node"""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def select_child(self, c_puct: float = 1.0) -> 'MCTSNode':
        """
        Select child with highest UCB score

        UCB = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        """
        best_score = -float('inf')
        best_child = None

        for child in self.children.values():
            # Q value
            q_value = child.value()

            # Exploration term
            u_value = c_puct * child.prior * math.sqrt(self.visit_count) / (1 + child.visit_count)

            score = q_value + u_value

            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def expand(self, policy: np.ndarray):
        """
        Expand node with children based on policy

        Args:
            policy: Move probabilities from neural network
        """
        legal_moves = self.game_state.get_legal_moves()
        board_size = self.game_state.board_size

        # Create child nodes for legal moves
        for move in legal_moves:
            row, col = move
            move_index = row * board_size + col

            # Create child game state
            child_game = self.game_state.clone()
            child_game.make_move(row, col)

            # Create child node with prior from policy
            prior = policy[move_index]
            self.children[move] = MCTSNode(child_game, parent=self, prior=prior, move=move)

        # Add pass move
        pass_index = board_size * board_size
        child_game = self.game_state.clone()
        child_game.pass_turn()
        self.children[None] = MCTSNode(child_game, parent=self, prior=policy[pass_index], move=None)

        self.is_expanded = True

    def backup(self, value: float):
        """
        Backup value through the tree
        Value is from perspective of player who played the move leading to this node
        """
        node = self
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            value = -value  # Flip value for opponent
            node = node.parent


class MCTS:
    """Monte Carlo Tree Search with neural network guidance"""

    def __init__(self, neural_net, num_simulations: int = 800, c_puct: float = 1.0):
        """
        Args:
            neural_net: Neural network for evaluating positions
            num_simulations: Number of MCTS simulations per move
            c_puct: Exploration constant
        """
        self.neural_net = neural_net
        self.num_simulations = num_simulations
        self.c_puct = c_puct

    def search(self, game_state: GoGame) -> Tuple[Optional[Tuple[int, int]], np.ndarray]:
        """
        Run MCTS search and return best move

        Args:
            game_state: Current game state

        Returns:
            best_move: (row, col) or None for pass
            move_probabilities: Distribution over moves
        """
        root = MCTSNode(game_state)

        # Run simulations
        for _ in range(self.num_simulations):
            node = root

            # Selection: traverse tree to leaf
            while not node.is_leaf() and not node.game_state.game_over:
                node = node.select_child(self.c_puct)

            # Expansion and evaluation
            if not node.game_state.game_over:
                # Get neural network predictions
                board = np.array([[cell.value for cell in row] for row in node.game_state.board])
                current_player = node.game_state.current_player.value

                policy, value = self.neural_net.predict(board, current_player)

                # Expand node
                node.expand(policy)

                # Backup value
                node.backup(value)
            else:
                # Terminal node - use actual game result
                score = node.game_state.get_score()
                # Determine winner from perspective of current player
                winner_value = 1.0 if score['black'] > score['white'] else -1.0
                # Flip if current player is white
                if node.game_state.current_player == Player.WHITE:
                    winner_value = -winner_value
                node.backup(winner_value)

        # Get visit counts for root children
        board_size = game_state.board_size
        visit_counts = np.zeros(board_size * board_size + 1)

        for move, child in root.children.items():
            if move is None:  # Pass
                visit_counts[board_size * board_size] = child.visit_count
            else:
                row, col = move
                visit_counts[row * board_size + col] = child.visit_count

        # Convert to probabilities
        if visit_counts.sum() > 0:
            move_probs = visit_counts / visit_counts.sum()
        else:
            move_probs = np.ones_like(visit_counts) / len(visit_counts)

        # Select move with highest visit count
        best_move_index = np.argmax(visit_counts)

        if best_move_index == board_size * board_size:
            best_move = None  # Pass
        else:
            row = best_move_index // board_size
            col = best_move_index % board_size
            best_move = (row, col)

        return best_move, move_probs

    def get_move_with_temp(self, game_state: GoGame, temperature: float = 1.0) -> Tuple[Optional[Tuple[int, int]], np.ndarray]:
        """
        Get move with temperature-based exploration

        Args:
            game_state: Current game state
            temperature: Temperature for move selection (higher = more exploration)
                        0 = greedy, 1 = proportional to visit counts

        Returns:
            selected_move: Chosen move
            move_probabilities: Distribution over moves
        """
        best_move, move_probs = self.search(game_state)

        if temperature == 0:
            return best_move, move_probs

        # Apply temperature
        visit_counts = move_probs * self.num_simulations
        visit_counts = visit_counts ** (1.0 / temperature)

        if visit_counts.sum() > 0:
            move_probs = visit_counts / visit_counts.sum()
        else:
            move_probs = np.ones_like(visit_counts) / len(visit_counts)

        # Sample move
        move_index = np.random.choice(len(move_probs), p=move_probs)

        board_size = game_state.board_size
        if move_index == board_size * board_size:
            selected_move = None
        else:
            row = move_index // board_size
            col = move_index % board_size
            selected_move = (row, col)

        return selected_move, move_probs
