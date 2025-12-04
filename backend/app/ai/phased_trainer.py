"""
Phased training system for AlphaGo Zero with checkpointing and game recording.
"""
import os
import json
import torch
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from pathlib import Path

from app.ai.neural_net import GoNet
from app.ai.mcts import MCTS
from app.game.go_game import GoGame, Player


class PhasedTrainer:
    """Enhanced trainer with phased training, checkpointing, and game recording."""

    def __init__(
        self,
        board_size: int = 9,
        num_channels: int = 128,
        num_res_blocks: int = 5,
        checkpoint_dir: str = "checkpoints",
        games_dir: str = "recorded_games"
    ):
        self.board_size = board_size
        self.neural_net = GoNet(
            board_size=board_size,
            num_channels=num_channels,
            num_res_blocks=num_res_blocks
        )
        self.optimizer = torch.optim.Adam(self.neural_net.parameters(), lr=0.001)

        # Training state
        self.current_episode = 0
        self.current_phase = 0
        self.is_training = False
        self.training_metrics = []

        # Directories for storage
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.games_dir = Path(games_dir)
        self.games_dir.mkdir(exist_ok=True)

        # Phase configuration
        self.phase_config = None

    def initialize_training(
        self,
        episodes_per_phase: int = 10000,
        checkpoint_interval: int = 1000,
        games_to_record_per_checkpoint: int = 3,
        num_simulations: int = 100
    ):
        """Initialize training configuration and save initial weights."""
        self.phase_config = {
            'episodes_per_phase': episodes_per_phase,
            'checkpoint_interval': checkpoint_interval,
            'games_to_record': games_to_record_per_checkpoint,
            'num_simulations': num_simulations
        }

        # Save initial weights (checkpoint 0)
        self.save_checkpoint(0, "initial")

        # Record initial self-play games
        print(f"Recording {games_to_record_per_checkpoint} initial self-play games...")
        self.record_self_play_games(0, games_to_record_per_checkpoint, num_simulations)

        print("Training initialized. Ready to start Phase 1.")

    def train_phase(
        self,
        phase_number: int,
        num_episodes: int,
        checkpoint_interval: int = 1000,
        num_simulations: int = 100,
        batch_size: int = 32
    ) -> Dict:
        """
        Train for a specific phase with automatic checkpointing.

        Args:
            phase_number: Current phase number (1, 2, 3, ...)
            num_episodes: Number of training episodes for this phase
            checkpoint_interval: Save checkpoint every N episodes
            num_simulations: MCTS simulations per move
            batch_size: Training batch size

        Returns:
            Dictionary with training metrics
        """
        self.current_phase = phase_number
        self.is_training = True
        phase_start_episode = self.current_episode

        print(f"\n=== Starting Phase {phase_number} ===")
        print(f"Training for {num_episodes} episodes")
        print(f"Checkpoints every {checkpoint_interval} episodes")

        training_data = []
        phase_metrics = {
            'phase': phase_number,
            'episodes': [],
            'losses': [],
            'policy_losses': [],
            'value_losses': []
        }

        for episode in range(num_episodes):
            if not self.is_training:
                print(f"Training paused at episode {self.current_episode}")
                break

            # Self-play game
            game_data = self.self_play_game(num_simulations)
            training_data.extend(game_data)

            # Train on accumulated data
            if len(training_data) >= batch_size:
                losses = self.train_on_batch(training_data[:batch_size])
                training_data = training_data[batch_size:]

                phase_metrics['episodes'].append(self.current_episode)
                phase_metrics['losses'].append(losses['total_loss'])
                phase_metrics['policy_losses'].append(losses['policy_loss'])
                phase_metrics['value_losses'].append(losses['value_loss'])

                self.training_metrics.append({
                    'episode': self.current_episode,
                    'phase': phase_number,
                    **losses
                })

            self.current_episode += 1

            # Checkpoint at intervals
            if (self.current_episode - phase_start_episode) % checkpoint_interval == 0:
                checkpoint_name = f"phase{phase_number}_ep{self.current_episode}"
                self.save_checkpoint(self.current_episode, checkpoint_name)

                # Record games at checkpoint
                print(f"\nRecording {self.phase_config['games_to_record']} games at checkpoint {self.current_episode}...")
                self.record_self_play_games(
                    self.current_episode,
                    self.phase_config['games_to_record'],
                    num_simulations
                )

            # Progress update
            if (episode + 1) % 100 == 0:
                avg_loss = np.mean(phase_metrics['losses'][-100:]) if phase_metrics['losses'] else 0
                print(f"Episode {self.current_episode} ({episode + 1}/{num_episodes}) - Avg Loss: {avg_loss:.4f}")

        # End of phase: record final games
        print(f"\n=== Phase {phase_number} Complete ===")
        print(f"Recording {self.phase_config['games_to_record']} final games for phase {phase_number}...")
        self.record_self_play_games(
            self.current_episode,
            self.phase_config['games_to_record'],
            num_simulations,
            label=f"phase{phase_number}_final"
        )

        # Save final checkpoint for this phase
        self.save_checkpoint(self.current_episode, f"phase{phase_number}_final")

        return phase_metrics

    def self_play_game(self, num_simulations: int = 100) -> List[Tuple]:
        """
        Play one self-play game and return training examples.

        Returns:
            List of (state, policy, value) tuples
        """
        game = GoGame(board_size=self.board_size)
        mcts = MCTS(self.neural_net, num_simulations=num_simulations)

        examples = []
        move_count = 0
        max_moves = self.board_size * self.board_size * 2  # Prevent infinite games

        while not game.game_over and move_count < max_moves:
            # Get move probabilities from MCTS
            move, move_probs = mcts.search(game)

            if move is None:
                # Pass move
                game.pass_turn()
            else:
                # Store state and policy for training
                state = self._encode_state(game)
                examples.append((state, move_probs, None))  # Value filled later

                # Make move
                row, col = move
                game.make_move(row, col)

            move_count += 1

        # Determine game outcome
        game.game_over = True
        black_score, white_score = game.calculate_score()

        if black_score > white_score:
            outcome = 1.0  # Black wins
        elif white_score > black_score:
            outcome = -1.0  # White wins
        else:
            outcome = 0.0  # Draw

        # Assign outcome to all positions
        training_examples = []
        for state, policy, _ in examples:
            # Flip outcome for white's perspective
            player = game.current_player
            value = outcome if player == Player.BLACK else -outcome
            training_examples.append((state, policy, value))

        return training_examples

    def record_self_play_games(
        self,
        checkpoint_episode: int,
        num_games: int,
        num_simulations: int,
        label: str = None
    ):
        """Record self-play games for visualization and analysis."""
        recorded_games = []

        for game_num in range(num_games):
            game = GoGame(board_size=self.board_size)
            mcts = MCTS(self.neural_net, num_simulations=num_simulations)

            game_record = {
                'checkpoint_episode': checkpoint_episode,
                'game_number': game_num + 1,
                'timestamp': datetime.now().isoformat(),
                'moves': [],
                'board_size': self.board_size
            }

            move_count = 0
            max_moves = self.board_size * self.board_size * 2

            while not game.game_over and move_count < max_moves:
                move, move_probs = mcts.search(game)

                if move is None:
                    game.pass_turn()
                    game_record['moves'].append({
                        'move_number': move_count + 1,
                        'player': 'black' if game.current_player == Player.BLACK else 'white',
                        'action': 'pass'
                    })
                else:
                    row, col = move
                    game_record['moves'].append({
                        'move_number': move_count + 1,
                        'player': 'black' if game.current_player == Player.BLACK else 'white',
                        'position': [row, col],
                        'board_state': [[cell.value for cell in board_row] for board_row in game.board]
                    })
                    game.make_move(row, col)

                move_count += 1

            # Final score
            game.game_over = True
            black_score, white_score = game.calculate_score()
            game_record['final_score'] = {
                'black': float(black_score),
                'white': float(white_score),
                'winner': 'black' if black_score > white_score else 'white' if white_score > black_score else 'draw'
            }

            recorded_games.append(game_record)
            print(f"  Game {game_num + 1}/{num_games} recorded - {game_record['final_score']['winner']} wins")

        # Save recorded games
        label_str = f"_{label}" if label else ""
        filename = self.games_dir / f"games_ep{checkpoint_episode}{label_str}.json"
        with open(filename, 'w') as f:
            # Convert any numpy types to Python native types for JSON serialization
            json.dump(recorded_games, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else x)

        print(f"Saved {num_games} games to {filename}")

    def train_on_batch(self, batch: List[Tuple]) -> Dict[str, float]:
        """Train neural network on a batch of examples."""
        states, policies, values = zip(*batch)

        states_tensor = torch.stack([torch.FloatTensor(s) for s in states])
        policies_tensor = torch.stack([torch.FloatTensor(p) for p in policies])
        values_tensor = torch.FloatTensor(values).unsqueeze(1)

        # Forward pass
        policy_logits, value_preds = self.neural_net(states_tensor)

        # Losses
        policy_loss = -torch.mean(torch.sum(policies_tensor * torch.log_softmax(policy_logits, dim=1), dim=1))
        value_loss = torch.mean((values_tensor - value_preds) ** 2)
        total_loss = policy_loss + value_loss

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item()
        }

    def save_checkpoint(self, episode: int, name: str):
        """Save model checkpoint with weights and metadata."""
        checkpoint_path = self.checkpoint_dir / f"{name}.pt"

        # Extract weight statistics for visualization
        weight_stats = {}
        for layer_name, param in self.neural_net.named_parameters():
            weight_stats[layer_name] = {
                'shape': list(param.shape),
                'mean': float(param.data.mean()),
                'std': float(param.data.std()),
                'min': float(param.data.min()),
                'max': float(param.data.max())
            }

        checkpoint = {
            'episode': episode,
            'phase': self.current_phase,
            'timestamp': datetime.now().isoformat(),
            'model_state_dict': self.neural_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'weight_stats': weight_stats,
            'training_metrics': self.training_metrics[-100:] if self.training_metrics else []
        }

        torch.save(checkpoint, checkpoint_path)

        # Also save metadata separately for easy access
        metadata_path = self.checkpoint_dir / f"{name}_metadata.json"
        metadata = {
            'episode': episode,
            'phase': self.current_phase,
            'timestamp': checkpoint['timestamp'],
            'weight_stats': weight_stats,
            'checkpoint_file': str(checkpoint_path)
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load a saved checkpoint."""
        checkpoint = torch.load(checkpoint_path)
        self.neural_net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_episode = checkpoint['episode']
        self.current_phase = checkpoint.get('phase', 0)
        print(f"Loaded checkpoint from episode {self.current_episode}")

    def pause_training(self):
        """Pause the current training phase."""
        self.is_training = False
        print(f"Training paused at episode {self.current_episode}")

    def resume_training(self):
        """Resume training."""
        self.is_training = True
        print(f"Training resumed from episode {self.current_episode}")

    def get_training_status(self) -> Dict:
        """Get current training status."""
        return {
            'is_training': self.is_training,
            'current_episode': self.current_episode,
            'current_phase': self.current_phase,
            'total_checkpoints': len(list(self.checkpoint_dir.glob('*.pt'))),
            'total_recorded_games': len(list(self.games_dir.glob('*.json'))),
            'recent_metrics': self.training_metrics[-10:] if self.training_metrics else []
        }

    def _encode_state(self, game: GoGame) -> np.ndarray:
        """
        Encode game state as neural network input.

        Returns:
            3D tensor (3, board_size, board_size) with:
            - Channel 0: Current player's stones
            - Channel 1: Opponent's stones
            - Channel 2: Current player indicator (all 1s for black, all 0s for white)
        """
        board_size = self.board_size
        state = np.zeros((3, board_size, board_size), dtype=np.float32)

        current_player = game.current_player
        opponent = Player.WHITE if current_player == Player.BLACK else Player.BLACK

        for row in range(board_size):
            for col in range(board_size):
                if game.board[row][col] == current_player:
                    state[0, row, col] = 1.0
                elif game.board[row][col] == opponent:
                    state[1, row, col] = 1.0

        # Current player indicator
        if current_player == Player.BLACK:
            state[2, :, :] = 1.0

        return state
