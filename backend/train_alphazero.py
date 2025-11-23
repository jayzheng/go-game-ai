"""
AlphaZero Training Script
Trains the Go AI using self-play reinforcement learning
"""
import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from app.api.training_routes import trainer
from app.ai.phased_trainer import PhasedTrainer


async def main():
    """
    Main training function
    Follows your specified workflow:
    1. Record 3 games at start
    2. Train for 10,000 games (Phase 1)
    3. Checkpoints every 1,000 games
    4. Record 3 games at each checkpoint
    5. Record 3 games at end
    """
    print("=" * 60)
    print("AlphaZero Go Training - Phase 1")
    print("=" * 60)
    print()

    # Initialize trainer
    print("Initializing training system...")
    global trainer
    trainer = PhasedTrainer(
        board_size=9,
        num_channels=128,
        num_res_blocks=5,
        checkpoint_dir="checkpoints",
        games_dir="recorded_games"
    )

    # Initialize and record 3 baseline games
    print("\nInitializing training configuration...")
    trainer.initialize_training(
        episodes_per_phase=10000,
        checkpoint_interval=1000,
        games_to_record_per_checkpoint=3,
        num_simulations=100  # MCTS simulations per move
    )

    print("\n" + "=" * 60)
    print("Starting Phase 1: 10,000 self-play games")
    print("=" * 60)
    print()
    print("Training Configuration:")
    print(f"  - Episodes: 10,000")
    print(f"  - Checkpoint interval: 1,000 episodes")
    print(f"  - Games recorded per checkpoint: 3")
    print(f"  - MCTS simulations per move: 100")
    print()

    # Start Phase 1 training
    phase_metrics = trainer.train_phase(
        phase_number=1,
        num_episodes=10000,
        checkpoint_interval=1000,
        num_simulations=100,
        batch_size=32
    )

    print("\n" + "=" * 60)
    print("Phase 1 Training Complete!")
    print("=" * 60)
    print()
    print(f"Total episodes completed: {trainer.current_episode}")
    print(f"Total checkpoints saved: {len(list(Path('checkpoints').glob('*.pt')))}")
    print(f"Total games recorded: {len(list(Path('recorded_games').glob('*.json')))}")
    print()
    print("Training metrics summary:")
    if phase_metrics['losses']:
        print(f"  - Average total loss: {sum(phase_metrics['losses']) / len(phase_metrics['losses']):.4f}")
        print(f"  - Average policy loss: {sum(phase_metrics['policy_losses']) / len(phase_metrics['policy_losses']):.4f}")
        print(f"  - Average value loss: {sum(phase_metrics['value_losses']) / len(phase_metrics['value_losses']):.4f}")
    print()
    print("Next steps:")
    print("  1. View training dashboard to analyze progress")
    print("  2. Compare weights across checkpoints")
    print("  3. Play against trained models to test strength")
    print("  4. Continue with Phase 2 if desired")
    print()


if __name__ == "__main__":
    asyncio.run(main())
