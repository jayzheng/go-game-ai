"""
API routes for testing trained AI models.
"""
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import torch

from app.ai.neural_net import GoNet
from app.ai.mcts import MCTS
from app.game.go_game import GoGame, Player


router = APIRouter(prefix="/api/test", tags=["testing"])

# Global test instances
test_games = {}
test_models = {}


class TestGameRequest(BaseModel):
    """Request to start a test game against a checkpoint."""
    checkpoint_name: str
    board_size: int = 9
    num_simulations: int = 100
    player_color: str = "black"  # "black" or "white"


class TestMoveRequest(BaseModel):
    """Request to make a move in a test game."""
    row: int
    col: int


@router.post("/game/new")
async def create_test_game(request: TestGameRequest):
    """
    Create a new test game against a specific checkpoint.
    Player can choose to play as black or white.
    """
    try:
        # Load the checkpoint
        checkpoint_dir = Path("checkpoints")
        checkpoint_path = checkpoint_dir / f"{request.checkpoint_name}.pt"

        if not checkpoint_path.exists():
            raise HTTPException(status_code=404, detail=f"Checkpoint '{request.checkpoint_name}' not found")

        # Load model
        if request.checkpoint_name not in test_models:
            checkpoint = torch.load(checkpoint_path)
            neural_net = GoNet(board_size=request.board_size)
            neural_net.load_state_dict(checkpoint['model_state_dict'])
            neural_net.eval()
            test_models[request.checkpoint_name] = neural_net

        # Create new game
        game = GoGame(board_size=request.board_size)
        game_id = f"test_{len(test_games)}_{request.checkpoint_name}"

        test_games[game_id] = {
            'game': game,
            'model': test_models[request.checkpoint_name],
            'num_simulations': request.num_simulations,
            'player_color': request.player_color,
            'checkpoint_name': request.checkpoint_name
        }

        # If player chose white, AI makes first move
        if request.player_color == "white":
            await _make_ai_move(game_id)

        return {
            'game_id': game_id,
            'checkpoint': request.checkpoint_name,
            'player_color': request.player_color,
            'board_size': request.board_size,
            'board': [[int(cell) for cell in row] for row in game.board],
            'current_player': int(game.current_player),
            'game_over': game.game_over
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create test game: {str(e)}")


@router.post("/game/{game_id}/move")
async def make_test_move(game_id: str, move_request: TestMoveRequest):
    """
    Make a move in a test game, then AI responds.
    """
    if game_id not in test_games:
        raise HTTPException(status_code=404, detail="Test game not found")

    game_data = test_games[game_id]
    game = game_data['game']

    if game.game_over:
        raise HTTPException(status_code=400, detail="Game is already over")

    # Player's move
    player_color = game_data['player_color']
    expected_player = Player.BLACK if player_color == "black" else Player.WHITE

    if game.current_player != expected_player:
        raise HTTPException(status_code=400, detail="Not player's turn")

    success = game.make_move(move_request.row, move_request.col)
    if not success:
        raise HTTPException(status_code=400, detail="Invalid move")

    # Check if game ended after player's move
    if game.game_over:
        return _get_game_state(game_id)

    # AI's response move
    await _make_ai_move(game_id)

    return _get_game_state(game_id)


@router.post("/game/{game_id}/pass")
async def pass_turn_test(game_id: str):
    """Pass turn in test game, then AI responds."""
    if game_id not in test_games:
        raise HTTPException(status_code=404, detail="Test game not found")

    game_data = test_games[game_id]
    game = game_data['game']

    if game.game_over:
        raise HTTPException(status_code=400, detail="Game is already over")

    game.pass_turn()

    # Check if game ended after pass
    if game.game_over:
        return _get_game_state(game_id)

    # AI's response
    await _make_ai_move(game_id)

    return _get_game_state(game_id)


@router.get("/game/{game_id}")
async def get_test_game_state(game_id: str):
    """Get current state of a test game."""
    if game_id not in test_games:
        raise HTTPException(status_code=404, detail="Test game not found")

    return _get_game_state(game_id)


@router.delete("/game/{game_id}")
async def end_test_game(game_id: str):
    """End and clean up a test game."""
    if game_id not in test_games:
        raise HTTPException(status_code=404, detail="Test game not found")

    del test_games[game_id]
    return {"message": "Test game ended"}


async def _make_ai_move(game_id: str):
    """Helper: Make AI move using MCTS."""
    game_data = test_games[game_id]
    game = game_data['game']
    model = game_data['model']
    num_simulations = game_data['num_simulations']

    mcts = MCTS(model, num_simulations=num_simulations)
    move, _ = mcts.search(game)

    if move is None:
        # AI passes
        game.pass_turn()
    else:
        row, col = move
        game.make_move(row, col)


def _get_game_state(game_id: str) -> dict:
    """Helper: Get game state as dict."""
    game_data = test_games[game_id]
    game = game_data['game']

    state = {
        'game_id': game_id,
        'checkpoint': game_data['checkpoint_name'],
        'player_color': game_data['player_color'],
        'board': [[int(cell) for cell in row] for row in game.board],
        'current_player': int(game.current_player),
        'game_over': game.game_over,
        'captured_stones': {
            'black': game.captured_stones[Player.BLACK],
            'white': game.captured_stones[Player.WHITE]
        },
        'move_history': [
            [row, col, int(player)]
            for row, col, player in game.move_history
        ]
    }

    if game.game_over:
        black_score, white_score = game.calculate_score()
        state['score'] = {
            'black': black_score,
            'white': white_score
        }

    return state
