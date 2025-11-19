import React, { useState, useEffect } from 'react';
import GoBoard from './components/GoBoard';
import GameInfo from './components/GameInfo';
import GameList from './components/GameList';
import { gameAPI } from './services/api';
import { GoEngine } from './utils/goEngine';
import './App.css';

function App() {
  const [gameId, setGameId] = useState(null);
  const [gameState, setGameState] = useState(null);
  const [legalMoves, setLegalMoves] = useState([]);
  const [showGameList, setShowGameList] = useState(false);
  const [message, setMessage] = useState('');
  const [isReplayMode, setIsReplayMode] = useState(false);
  const [replayMoves, setReplayMoves] = useState([]);
  const [replayIndex, setReplayIndex] = useState(0);
  const [replayEngine, setReplayEngine] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);

  useEffect(() => {
    startNewGame();
  }, []);

  const startNewGame = async () => {
    try {
      setMessage('Starting new game...');
      const data = await gameAPI.createNewGame(9);
      setGameId(data.game_id);
      await loadGameState(data.game_id);
      setIsReplayMode(false);
      setReplayMoves([]);
      setReplayIndex(0);
      setMessage('New game started!');
      setTimeout(() => setMessage(''), 2000);
    } catch (error) {
      console.error('Failed to create game:', error);
      setMessage('Failed to create game. Make sure backend is running.');
    }
  };

  const loadGameState = async (gId) => {
    try {
      const state = await gameAPI.getGameState(gId);
      setGameState(state);

      if (!state.game_over && state.current_player === 1) {
        const moves = await gameAPI.getLegalMoves(gId);
        setLegalMoves(moves.legal_moves || []);
      } else {
        setLegalMoves([]);
      }
    } catch (error) {
      console.error('Failed to load game state:', error);
    }
  };

  const handleCellClick = async (row, col) => {
    if (!gameId || !gameState || gameState.game_over || isReplayMode) return;
    if (gameState.current_player !== 1) return; // Not player's turn
    if (isProcessing) return; // Prevent multiple clicks

    // Check if position is already occupied
    if (gameState.board[row][col] !== 0) return;

    setIsProcessing(true);

    try {
      const newState = await gameAPI.makeMove(gameId, row, col);
      setGameState(newState);

      if (!newState.game_over && newState.current_player === 1) {
        const moves = await gameAPI.getLegalMoves(gameId);
        setLegalMoves(moves.legal_moves || []);
      } else {
        setLegalMoves([]);
      }
    } catch (error) {
      console.error('Failed to make move:', error);
      setMessage('Invalid move!');
      setTimeout(() => setMessage(''), 2000);
    } finally {
      setIsProcessing(false);
    }
  };

  const handlePass = async () => {
    if (!gameId || !gameState || gameState.game_over || isReplayMode) return;

    try {
      const newState = await gameAPI.passTurn(gameId);
      setGameState(newState);

      if (!newState.game_over && newState.current_player === 1) {
        const moves = await gameAPI.getLegalMoves(gameId);
        setLegalMoves(moves.legal_moves || []);
      } else {
        setLegalMoves([]);
      }
    } catch (error) {
      console.error('Failed to pass:', error);
    }
  };

  const handleSave = async () => {
    if (!gameId) return;

    try {
      await gameAPI.saveGame(gameId);
      setMessage('Game saved successfully!');
      setTimeout(() => setMessage(''), 2000);
    } catch (error) {
      console.error('Failed to save game:', error);
      setMessage('Failed to save game.');
      setTimeout(() => setMessage(''), 2000);
    }
  };

  const handleReplay = async (dbId) => {
    try {
      const replayData = await gameAPI.getGameReplay(dbId);
      setShowGameList(false);
      setIsReplayMode(true);
      setReplayMoves(replayData.move_history);
      setReplayIndex(0);

      // Initialize game engine
      const boardSize = replayData.board_size;
      const engine = new GoEngine(boardSize);
      setReplayEngine(engine);

      setGameState({
        board: engine.getBoard(),
        current_player: 1,
        captured_stones: { black: 0, white: 0 },
        game_over: false,
        score: null,
        move_history: []
      });

      setMessage('Replay mode - Click Next Move to view game');
    } catch (error) {
      console.error('Failed to load replay:', error);
      setMessage('Failed to load replay.');
    }
  };

  const handleNextMove = () => {
    if (!isReplayMode || replayIndex >= replayMoves.length || !replayEngine) return;

    const move = replayMoves[replayIndex];
    const [row, col, player] = move;

    if (row === -1 && col === -1) {
      // Pass move
      setReplayIndex(replayIndex + 1);
      return;
    }

    // Apply move with game rules (including captures)
    replayEngine.makeMove(row, col, player);

    setGameState({
      ...gameState,
      board: replayEngine.getBoard(),
      current_player: player === 1 ? 2 : 1
    });

    setReplayIndex(replayIndex + 1);
  };

  const handlePrevMove = () => {
    if (!isReplayMode || replayIndex <= 0) return;

    // Rebuild board from scratch up to previous move
    const boardSize = gameState.board.length;
    const engine = new GoEngine(boardSize);

    for (let i = 0; i < replayIndex - 1; i++) {
      const [row, col, player] = replayMoves[i];
      if (row !== -1 && col !== -1) {
        engine.makeMove(row, col, player);
      }
    }

    setReplayEngine(engine);
    setGameState({
      ...gameState,
      board: engine.getBoard()
    });

    setReplayIndex(replayIndex - 1);
  };

  if (!gameState) {
    return (
      <div className="app">
        <div className="loading">Loading game...</div>
      </div>
    );
  }

  return (
    <div className="app">
      {message && <div className="message">{message}</div>}

      <div className="game-container">
        <GoBoard
          board={gameState.board}
          onCellClick={handleCellClick}
          legalMoves={legalMoves}
          currentPlayer={gameState.current_player}
          isProcessing={isProcessing}
        />

        <GameInfo
          currentPlayer={gameState.current_player}
          capturedStones={gameState.captured_stones}
          gameOver={gameState.game_over}
          score={gameState.score}
          onNewGame={startNewGame}
          onPass={handlePass}
          onSave={handleSave}
          onViewGames={() => setShowGameList(true)}
        />
      </div>

      {isReplayMode && (
        <div className="replay-controls">
          <button onClick={handlePrevMove} disabled={replayIndex <= 0}>
            Previous Move
          </button>
          <span>Move {replayIndex} / {replayMoves.length}</span>
          <button onClick={handleNextMove} disabled={replayIndex >= replayMoves.length}>
            Next Move
          </button>
          <button onClick={() => {
            setIsReplayMode(false);
            startNewGame();
          }}>
            Exit Replay
          </button>
        </div>
      )}

      {showGameList && (
        <GameList
          onClose={() => setShowGameList(false)}
          onReplay={handleReplay}
        />
      )}
    </div>
  );
}

export default App;
