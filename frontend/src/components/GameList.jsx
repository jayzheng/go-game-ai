import React, { useState, useEffect } from 'react';
import { gameAPI } from '../services/api';
import './GameList.css';

const GameList = ({ onClose, onReplay }) => {
  const [games, setGames] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadGames();
  }, []);

  const loadGames = async () => {
    try {
      const data = await gameAPI.listGames();
      setGames(data);
    } catch (error) {
      console.error('Failed to load games:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h2>Saved Games</h2>
          <button className="close-btn" onClick={onClose}>&times;</button>
        </div>

        <div className="games-list">
          {loading ? (
            <p>Loading games...</p>
          ) : games.length === 0 ? (
            <p>No saved games yet.</p>
          ) : (
            games.map((game) => (
              <div key={game.id} className="game-item">
                <div className="game-info-text">
                  <p><strong>Date:</strong> {new Date(game.created_at).toLocaleString()}</p>
                  <p><strong>Players:</strong> {game.black_player} vs {game.white_player}</p>
                  <p><strong>Status:</strong> {game.status}</p>
                  {game.winner && (
                    <>
                      <p><strong>Winner:</strong> {game.winner}</p>
                      <p><strong>Score:</strong> Black {game.black_score} - White {game.white_score}</p>
                    </>
                  )}
                </div>
                <button
                  className="btn btn-small"
                  onClick={() => onReplay(game.id)}
                >
                  Replay
                </button>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
};

export default GameList;
