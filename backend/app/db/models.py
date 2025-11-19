"""
Database models for storing games and game history
"""
from sqlalchemy import Column, Integer, String, DateTime, Text, Float
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()


class Game(Base):
    __tablename__ = "games"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    black_player = Column(String, default="Human")
    white_player = Column(String, default="AI")
    board_size = Column(Integer, default=9)
    move_history = Column(Text)  # JSON string of moves
    final_score = Column(String, nullable=True)  # JSON string of final score
    winner = Column(String, nullable=True)
    game_status = Column(String, default="in_progress")  # in_progress, completed
    black_score = Column(Float, nullable=True)
    white_score = Column(Float, nullable=True)

    def __repr__(self):
        return f"<Game(id={self.id}, black={self.black_player}, white={self.white_player}, status={self.game_status})>"
