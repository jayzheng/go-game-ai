import axios from 'axios';

const API_BASE_URL = '/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const gameAPI = {
  createNewGame: async (boardSize = 9) => {
    const response = await api.post('/game/new', {
      board_size: boardSize,
      black_player: 'Human',
      white_player: 'AI',
    });
    return response.data;
  },

  getGameState: async (gameId) => {
    const response = await api.get(`/game/${gameId}`);
    return response.data;
  },

  makeMove: async (gameId, row, col) => {
    const response = await api.post('/game/move', {
      game_id: gameId,
      row,
      col,
    });
    return response.data;
  },

  passTurn: async (gameId) => {
    const response = await api.post('/game/pass', {
      game_id: gameId,
    });
    return response.data;
  },

  getLegalMoves: async (gameId) => {
    const response = await api.get(`/game/${gameId}/legal-moves`);
    return response.data;
  },

  saveGame: async (gameId) => {
    const response = await api.post(`/game/${gameId}/save`);
    return response.data;
  },

  listGames: async () => {
    const response = await api.get('/games');
    return response.data;
  },

  getGameReplay: async (dbId) => {
    const response = await api.get(`/game/replay/${dbId}`);
    return response.data;
  },
};

export default api;
