#!/usr/bin/env python
"""
Training script for AlphaGo Zero model

Usage:
    python train.py --iterations 10 --games 100 --simulations 200
"""
import argparse
from app.ai.training import AlphaZeroTrainer


def main():
    parser = argparse.ArgumentParser(description='Train AlphaGo Zero model')
    parser.add_argument('--iterations', type=int, default=10,
                       help='Number of training iterations')
    parser.add_argument('--games', type=int, default=100,
                       help='Number of self-play games per iteration')
    parser.add_argument('--simulations', type=int, default=100,
                       help='Number of MCTS simulations per move')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs per iteration')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Training batch size')
    parser.add_argument('--board-size', type=int, default=9,
                       help='Board size (9, 13, or 19)')
    parser.add_argument('--model-dir', type=str, default='./models',
                       help='Directory to save models')

    args = parser.parse_args()

    print("="*70)
    print("AlphaGo Zero Training")
    print("="*70)
    print(f"Board size: {args.board_size}x{args.board_size}")
    print(f"Iterations: {args.iterations}")
    print(f"Games per iteration: {args.games}")
    print(f"MCTS simulations: {args.simulations}")
    print(f"Training epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print("="*70)
    print()

    # Create trainer
    trainer = AlphaZeroTrainer(
        board_size=args.board_size,
        model_dir=args.model_dir
    )

    # Run training iterations
    for iteration in range(args.iterations):
        print(f"\n{'='*70}")
        print(f"ITERATION {iteration + 1}/{args.iterations}")
        print(f"{'='*70}\n")

        trainer.train_iteration(
            num_games=args.games,
            num_simulations=args.simulations,
            epochs=args.epochs,
            batch_size=args.batch_size
        )

    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    print(f"Models saved in: {args.model_dir}")


if __name__ == "__main__":
    main()
