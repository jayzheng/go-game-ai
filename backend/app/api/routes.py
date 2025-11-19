"""
FastAPI routes for Go game API
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel
from typing import List, Optional
import json

from ..game.go_game import GoGame, Player
from ..ai.random_player import RandomPlayer
from ..db.database import get_session
from ..db.models import Game

router = APIRouter()

# Store active games in memory (in production, use Redis or similar)
active_games = {}


class NewGameRequest(BaseModel):
    board_size: int = 9
    black_player: str = "Human"
    white_player: str = "AI"


class MoveRequest(BaseModel):
    game_id: str
    row: int
    col: int


class PassRequest(BaseModel):
    game_id: str


class GameResponse(BaseModel):
    game_id: str
    board: List[List[int]]
    current_player: int
    captured_stones: dict
    game_over: bool
    score: Optional[dict]
    move_history: List[tuple]


@router.post("/game/new", response_model=dict)
async def create_new_game(request: NewGameRequest):
    """Create a new game (in memory only, not saved to database)"""
    game = GoGame(board_size=request.board_size)
    game_id = f"game_{len(active_games) + 1}"
    active_games[game_id] = {
        'game': game,
        'black_player': request.black_player,
        'white_player': request.white_player
    }

    return {
        "game_id": game_id,
        "message": "Game created successfully"
    }


@router.get("/game/{game_id}", response_model=GameResponse)
async def get_game_state(game_id: str):
    """Get current game state"""
    if game_id not in active_games:
        raise HTTPException(status_code=404, detail="Game not found")

    game = active_games[game_id]['game']
    state = game.get_state()

    return GameResponse(
        game_id=game_id,
        board=state['board'],
        current_player=state['current_player'],
        captured_stones=state['captured_stones'],
        game_over=state['game_over'],
        score=state['score'],
        move_history=state['move_history']
    )


@router.post("/game/move", response_model=GameResponse)
async def make_move(request: MoveRequest):
    """Make a move in the game"""
    if request.game_id not in active_games:
        raise HTTPException(status_code=404, detail="Game not found")

    game = active_games[request.game_id]['game']

    # Make player move
    success = game.make_move(request.row, request.col)
    if not success:
        raise HTTPException(status_code=400, detail="Invalid move")

    # If game is not over and current player is AI, make AI move
    if not game.game_over and game.current_player == Player.WHITE:
        ai = RandomPlayer()
        ai_move = ai.get_move(game)

        if ai_move is None:
            game.pass_turn()
        else:
            game.make_move(ai_move[0], ai_move[1])

    state = game.get_state()
    return GameResponse(
        game_id=request.game_id,
        board=state['board'],
        current_player=state['current_player'],
        captured_stones=state['captured_stones'],
        game_over=state['game_over'],
        score=state['score'],
        move_history=state['move_history']
    )


@router.post("/game/pass", response_model=GameResponse)
async def pass_turn(request: PassRequest):
    """Pass the turn"""
    if request.game_id not in active_games:
        raise HTTPException(status_code=404, detail="Game not found")

    game = active_games[request.game_id]['game']
    game.pass_turn()

    # If game is not over and current player is AI, make AI move
    if not game.game_over and game.current_player == Player.WHITE:
        ai = RandomPlayer()
        ai_move = ai.get_move(game)

        if ai_move is None:
            game.pass_turn()
        else:
            game.make_move(ai_move[0], ai_move[1])

    state = game.get_state()
    return GameResponse(
        game_id=request.game_id,
        board=state['board'],
        current_player=state['current_player'],
        captured_stones=state['captured_stones'],
        game_over=state['game_over'],
        score=state['score'],
        move_history=state['move_history']
    )


@router.get("/game/{game_id}/legal-moves")
async def get_legal_moves(game_id: str):
    """Get all legal moves for current player"""
    if game_id not in active_games:
        raise HTTPException(status_code=404, detail="Game not found")

    game = active_games[game_id]['game']
    legal_moves = game.get_legal_moves()

    return {"legal_moves": legal_moves}


@router.post("/game/{game_id}/save")
async def save_game(game_id: str, session: AsyncSession = Depends(get_session)):
    """Save game to database"""
    if game_id not in active_games:
        raise HTTPException(status_code=404, detail="Game not found")

    game_data = active_games[game_id]
    game = game_data['game']
    state = game.get_state()

    # Create new database record
    db_game = Game(
        black_player=game_data['black_player'],
        white_player=game_data['white_player'],
        board_size=game.board_size,
        move_history=json.dumps(state['move_history']),
        game_status="completed" if state['game_over'] else "in_progress"
    )

    if state['game_over'] and state['score']:
        db_game.black_score = state['score']['black']
        db_game.white_score = state['score']['white']
        db_game.final_score = json.dumps(state['score'])
        db_game.winner = "black" if state['score']['black'] > state['score']['white'] else "white"

    session.add(db_game)
    await session.commit()

    return {"message": "Game saved successfully"}


@router.get("/games", response_model=List[dict])
async def list_games(session: AsyncSession = Depends(get_session)):
    """List all saved games"""
    result = await session.execute(select(Game).order_by(Game.created_at.desc()))
    games = result.scalars().all()

    return [
        {
            "id": game.id,
            "created_at": game.created_at.isoformat(),
            "black_player": game.black_player,
            "white_player": game.white_player,
            "status": game.game_status,
            "winner": game.winner,
            "black_score": game.black_score,
            "white_score": game.white_score
        }
        for game in games
    ]


@router.get("/game/replay/{db_id}")
async def get_game_replay(db_id: int, session: AsyncSession = Depends(get_session)):
    """Get game for replay"""
    result = await session.execute(select(Game).where(Game.id == db_id))
    game = result.scalar_one_or_none()

    if not game:
        raise HTTPException(status_code=404, detail="Game not found in database")

    return {
        "id": game.id,
        "created_at": game.created_at.isoformat(),
        "black_player": game.black_player,
        "white_player": game.white_player,
        "board_size": game.board_size,
        "move_history": json.loads(game.move_history),
        "final_score": json.loads(game.final_score) if game.final_score else None,
        "winner": game.winner
    }
