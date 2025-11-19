#!/usr/bin/env python
"""
Test trained model by playing a game

Usage:
    python test_model.py --model models/model_20240101_120000.pt
"""
import argparse
from app.game.go_game import GoGame, Player
from app.ai.alphazero_player import AlphaZeroPlayer
from app.ai.random_player import RandomPlayer


def display_board(game: GoGame):
    """Display the current board state"""
    print("\n   ", end="")
    for i in range(game.board_size):
        print(f"{i:2}", end=" ")
    print()

    for i, row in enumerate(game.board):
        print(f"{i:2} ", end="")
        for cell in row:
            if cell == Player.BLACK:
                print(" X", end=" ")
            elif cell == Player.WHITE:
                print(" O", end=" ")
            else:
                print(" .", end=" ")
        print()


def play_game(model_path: str = None, simulations: int = 100, board_size: int = 9):
    """Play a game between AlphaZero and Random player"""

    print("\n" + "="*70)
    print("Testing AlphaGo Zero Model")
    print("="*70)

    game = GoGame(board_size=board_size)

    # Create players
    if model_path:
        black_player = AlphaZeroPlayer(model_path=model_path, num_simulations=simulations)
        print(f"Black: AlphaZero (loaded from {model_path})")
    else:
        black_player = AlphaZeroPlayer(num_simulations=simulations)
        print("Black: AlphaZero (random initialization)")

    white_player = RandomPlayer()
    print("White: Random Player")
    print("="*70)

    move_count = 0
    max_moves = board_size * board_size * 2

    while not game.game_over and move_count < max_moves:
        display_board(game)

        current_player_name = "Black (AlphaZero)" if game.current_player == Player.BLACK else "White (Random)"
        print(f"\nMove {move_count + 1}: {current_player_name}")

        # Get move from appropriate player
        if game.current_player == Player.BLACK:
            move = black_player.get_move(game)
        else:
            move = white_player.get_move(game)

        # Make move
        if move is None:
            print("Pass")
            game.pass_turn()
        else:
            row, col = move
            print(f"Play at ({row}, {col})")
            success = game.make_move(row, col)
            if not success:
                print("Invalid move! Passing instead.")
                game.pass_turn()

        move_count += 1

    # Game over
    display_board(game)
    print("\n" + "="*70)
    print("Game Over!")
    print("="*70)

    score = game.get_score()
    print(f"\nFinal Score:")
    print(f"Black: {score['black']} points")
    print(f"  - Stones: {score['black_stones']}")
    print(f"  - Territory: {score['black_territory']}")
    print(f"  - Captures: {score['black_captures']}")
    print(f"\nWhite: {score['white']} points")
    print(f"  - Stones: {score['white_stones']}")
    print(f"  - Territory: {score['white_territory']}")
    print(f"  - Captures: {score['white_captures']}")

    winner = "Black (AlphaZero)" if score['black'] > score['white'] else "White (Random)"
    print(f"\nWinner: {winner}")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description='Test trained AlphaGo Zero model')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to trained model checkpoint')
    parser.add_argument('--simulations', type=int, default=100,
                       help='Number of MCTS simulations per move')
    parser.add_argument('--board-size', type=int, default=9,
                       help='Board size (9, 13, or 19)')

    args = parser.parse_args()

    play_game(
        model_path=args.model,
        simulations=args.simulations,
        board_size=args.board_size
    )


if __name__ == "__main__":
    main()
