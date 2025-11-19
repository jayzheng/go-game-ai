import React from 'react';
import './GameInfo.css';

const GameInfo = ({
  currentPlayer,
  capturedStones,
  gameOver,
  score,
  onNewGame,
  onPass,
  onSave,
  onViewGames
}) => {
  const playerName = currentPlayer === 1 ? 'Black' : 'White';

  return (
    <div className="game-info">
      <h1>Go AI Game</h1>

      <div className="status">
        {!gameOver ? (
          <div className="current-turn">
            <div className={`player-indicator ${playerName.toLowerCase()}`}></div>
            <span>Current Turn: {playerName}</span>
          </div>
        ) : (
          <div className="game-over">
            <h2>Game Over!</h2>
            {score && (
              <div className="final-score">
                <p><strong>Black:</strong> {score.black} points</p>
                <p><strong>White:</strong> {score.white} points</p>
                <p className="winner">
                  Winner: {score.black > score.white ? 'Black' : 'White'}
                </p>
              </div>
            )}
          </div>
        )}
      </div>

      <div className="captures">
        <h3>Captured Stones</h3>
        <div className="capture-info">
          <div className="capture-item">
            <div className="stone black"></div>
            <span>Black: {capturedStones.black || 0}</span>
          </div>
          <div className="capture-item">
            <div className="stone white"></div>
            <span>White: {capturedStones.white || 0}</span>
          </div>
        </div>
      </div>

      <div className="controls">
        <button onClick={onNewGame} className="btn btn-primary">
          New Game
        </button>
        {!gameOver && (
          <button onClick={onPass} className="btn btn-secondary">
            Pass Turn
          </button>
        )}
        <button onClick={onSave} className="btn btn-success">
          Save Game
        </button>
        <button onClick={onViewGames} className="btn btn-info">
          View Saved Games
        </button>
      </div>
    </div>
  );
};

export default GameInfo;
