"""
Self-play training pipeline for AlphaGo Zero
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple
import os
from datetime import datetime

from .neural_net import GoNet
from .mcts import MCTS
from ..game.go_game import GoGame, Player


class GameDataset(Dataset):
    """Dataset for training from self-play games"""

    def __init__(self, examples: List[Tuple[np.ndarray, np.ndarray, float]]):
        """
        Args:
            examples: List of (board_state, policy, value) tuples
        """
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        state, policy, value = self.examples[idx]
        return (
            torch.from_numpy(state).float(),
            torch.from_numpy(policy).float(),
            torch.tensor([value]).float()
        )


class AlphaZeroTrainer:
    """Trainer for AlphaGo Zero style learning"""

    def __init__(self, board_size: int = 9, model_dir: str = './models'):
        self.board_size = board_size
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

        # Initialize neural network
        self.neural_net = GoNet(board_size=board_size)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.neural_net.to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(self.neural_net.parameters(), lr=0.001, weight_decay=1e-4)

        # Training examples buffer
        self.training_examples = []
        self.max_examples = 100000

    def self_play_game(self, num_simulations: int = 100, temperature: float = 1.0) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """
        Play one game through self-play

        Args:
            num_simulations: Number of MCTS simulations per move
            temperature: Temperature for move selection

        Returns:
            List of training examples (board_state, policy, value)
        """
        game = GoGame(board_size=self.board_size)
        mcts = MCTS(self.neural_net, num_simulations=num_simulations)

        examples = []
        move_count = 0

        while not game.game_over and move_count < self.board_size * self.board_size * 2:
            # Use temperature for first 30 moves, then greedy
            temp = temperature if move_count < 30 else 0.1

            # Get move from MCTS
            move, move_probs = mcts.get_move_with_temp(game, temperature=temp)

            # Record training example
            board = np.array([[cell.value for cell in row] for row in game.board])
            current_player = game.current_player.value

            # Convert board to input tensor format
            state_tensor = self.neural_net.board_to_tensor(board, current_player).numpy()
            examples.append((state_tensor, move_probs, None))  # Value will be filled later

            # Make move
            if move is None:
                game.pass_turn()
            else:
                game.make_move(move[0], move[1])

            move_count += 1

        # Get final game result
        score = game.get_score()
        winner = 1.0 if score['black'] > score['white'] else -1.0

        # Fill in values based on game outcome
        final_examples = []
        for i, (state, policy, _) in enumerate(examples):
            # Value is from perspective of player who made the move
            # Alternate between black and white
            player_perspective = 1 if i % 2 == 0 else -1  # Black on even moves, white on odd
            value = winner * player_perspective
            final_examples.append((state, policy, value))

        return final_examples

    def generate_self_play_data(self, num_games: int = 100, num_simulations: int = 100):
        """
        Generate training data through self-play

        Args:
            num_games: Number of self-play games
            num_simulations: MCTS simulations per move
        """
        print(f"Generating {num_games} self-play games...")

        for i in range(num_games):
            examples = self.self_play_game(num_simulations=num_simulations)
            self.training_examples.extend(examples)

            if (i + 1) % 10 == 0:
                print(f"Completed {i + 1}/{num_games} games, {len(self.training_examples)} examples")

        # Limit buffer size
        if len(self.training_examples) > self.max_examples:
            self.training_examples = self.training_examples[-self.max_examples:]

        print(f"Self-play complete. Total examples: {len(self.training_examples)}")

    def train(self, epochs: int = 10, batch_size: int = 32):
        """
        Train neural network on self-play data

        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        if not self.training_examples:
            print("No training examples available. Generate self-play data first.")
            return

        dataset = GameDataset(self.training_examples)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.neural_net.train()

        for epoch in range(epochs):
            total_loss = 0.0
            policy_loss_total = 0.0
            value_loss_total = 0.0

            for batch_idx, (states, target_policies, target_values) in enumerate(dataloader):
                states = states.to(self.device)
                target_policies = target_policies.to(self.device)
                target_values = target_values.to(self.device)

                # Forward pass
                pred_policies, pred_values = self.neural_net(states)

                # Compute losses
                policy_loss = -torch.mean(torch.sum(target_policies * pred_policies, dim=1))
                value_loss = nn.MSELoss()(pred_values, target_values)
                loss = policy_loss + value_loss

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                policy_loss_total += policy_loss.item()
                value_loss_total += value_loss.item()

            avg_loss = total_loss / len(dataloader)
            avg_policy_loss = policy_loss_total / len(dataloader)
            avg_value_loss = value_loss_total / len(dataloader)

            print(f"Epoch {epoch + 1}/{epochs}: Loss={avg_loss:.4f}, "
                  f"Policy Loss={avg_policy_loss:.4f}, Value Loss={avg_value_loss:.4f}")

    def save_model(self, name: str = None):
        """Save current model"""
        if name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name = f"model_{timestamp}.pt"

        path = os.path.join(self.model_dir, name)
        self.neural_net.save_checkpoint(path, self.optimizer)
        print(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load model from checkpoint"""
        self.neural_net, _ = GoNet.load_checkpoint(path, self.optimizer)
        self.neural_net.to(self.device)
        print(f"Model loaded from {path}")

    def train_iteration(self, num_games: int = 100, num_simulations: int = 100,
                       epochs: int = 10, batch_size: int = 32):
        """
        Complete training iteration: self-play + training

        Args:
            num_games: Number of self-play games
            num_simulations: MCTS simulations per move
            epochs: Training epochs
            batch_size: Batch size
        """
        print(f"\n{'='*60}")
        print(f"Training Iteration")
        print(f"{'='*60}\n")

        # Generate self-play data
        self.generate_self_play_data(num_games, num_simulations)

        # Train on data
        self.train(epochs, batch_size)

        # Save model
        self.save_model()

        print(f"\n{'='*60}")
        print(f"Iteration Complete")
        print(f"{'='*60}\n")


def main():
    """Example training script"""
    print("AlphaGo Zero Training Pipeline")
    print("================================\n")

    # Create trainer
    trainer = AlphaZeroTrainer(board_size=9)

    # Run training iterations
    num_iterations = 10

    for iteration in range(num_iterations):
        print(f"\n\nIteration {iteration + 1}/{num_iterations}")
        trainer.train_iteration(
            num_games=10,      # Start with fewer games for testing
            num_simulations=50,  # Fewer simulations for faster training
            epochs=5,
            batch_size=32
        )


if __name__ == "__main__":
    main()
