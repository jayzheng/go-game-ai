/**
 * Client-side Go game engine for replay functionality
 * Implements basic capture rules
 */

export class GoEngine {
  constructor(boardSize = 9) {
    this.boardSize = boardSize;
    this.board = Array(boardSize).fill(null).map(() => Array(boardSize).fill(0));
  }

  makeMove(row, col, player) {
    // Place stone
    this.board[row][col] = player;

    // Check for captures of opponent stones
    const opponent = player === 1 ? 2 : 1;
    this.removeCapturedStones(opponent);

    return this.board;
  }

  getNeighbors(row, col) {
    const neighbors = [];
    const directions = [[-1, 0], [1, 0], [0, -1], [0, 1]];

    for (const [dr, dc] of directions) {
      const newRow = row + dr;
      const newCol = col + dc;
      if (newRow >= 0 && newRow < this.boardSize && newCol >= 0 && newCol < this.boardSize) {
        neighbors.push([newRow, newCol]);
      }
    }

    return neighbors;
  }

  getGroup(row, col) {
    const color = this.board[row][col];
    if (color === 0) return new Set();

    const group = new Set();
    const stack = [[row, col]];
    const visited = new Set();

    while (stack.length > 0) {
      const [r, c] = stack.pop();
      const key = `${r},${c}`;

      if (visited.has(key)) continue;
      visited.add(key);

      if (this.board[r][c] === color) {
        group.add(key);

        for (const [nr, nc] of this.getNeighbors(r, c)) {
          if (!visited.has(`${nr},${nc}`)) {
            stack.push([nr, nc]);
          }
        }
      }
    }

    return group;
  }

  hasLiberties(group) {
    for (const posKey of group) {
      const [row, col] = posKey.split(',').map(Number);

      for (const [nr, nc] of this.getNeighbors(row, col)) {
        if (this.board[nr][nc] === 0) {
          return true;
        }
      }
    }

    return false;
  }

  removeCapturedStones(player) {
    const toRemove = [];

    for (let row = 0; row < this.boardSize; row++) {
      for (let col = 0; col < this.boardSize; col++) {
        if (this.board[row][col] === player) {
          const group = this.getGroup(row, col);
          if (!this.hasLiberties(group)) {
            toRemove.push(...group);
          }
        }
      }
    }

    // Remove captured stones
    const removed = new Set();
    for (const posKey of toRemove) {
      if (!removed.has(posKey)) {
        const [row, col] = posKey.split(',').map(Number);
        this.board[row][col] = 0;
        removed.add(posKey);
      }
    }
  }

  getBoard() {
    return this.board.map(row => [...row]);
  }

  clone() {
    const newEngine = new GoEngine(this.boardSize);
    newEngine.board = this.board.map(row => [...row]);
    return newEngine;
  }
}
