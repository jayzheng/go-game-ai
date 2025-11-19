import React from 'react';
import './GoBoard.css';

const GoBoard = ({ board, onCellClick, legalMoves = [], currentPlayer, isProcessing = false }) => {
  const boardSize = board.length;

  const isLegalMove = (row, col) => {
    return legalMoves.some(([r, c]) => r === row && c === col);
  };

  const getCellContent = (value) => {
    if (value === 1) return 'black';
    if (value === 2) return 'white';
    return null;
  };

  return (
    <div className="go-board-container">
      <div className="go-board" style={{
        gridTemplateColumns: `repeat(${boardSize}, 1fr)`,
        gridTemplateRows: `repeat(${boardSize}, 1fr)`,
        opacity: isProcessing ? 0.6 : 1,
        pointerEvents: isProcessing ? 'none' : 'auto'
      }}>
        {board.map((row, rowIndex) =>
          row.map((cell, colIndex) => {
            const stone = getCellContent(cell);
            const isLegal = isLegalMove(rowIndex, colIndex);
            const isEmpty = cell === 0;

            return (
              <div
                key={`${rowIndex}-${colIndex}`}
                className={`intersection ${stone ? 'has-stone' : ''} ${isEmpty && !isProcessing ? 'clickable' : ''}`}
                onClick={() => !isProcessing && isEmpty && currentPlayer === 1 && onCellClick(rowIndex, colIndex)}
              >
                {stone && <div className={`stone ${stone}`} />}
              </div>
            );
          })
        )}
      </div>
    </div>
  );
};

export default GoBoard;
