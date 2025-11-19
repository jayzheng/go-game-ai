# Go AI Game - AlphaGo Zero Style

A full-stack Go game application with AI opponent, built with React frontend and Python (PyTorch + FastAPI) backend.

## Features

- **9x9 Go Board** - Fast training and gameplay
- **AI Opponent** - Currently random moves (AlphaGo Zero implementation in progress)
- **Game Storage** - Save and replay your games
- **Web UI** - React-based beautiful interface
- **Mobile Ready** - React Native app (coming soon)

## Project Structure

```
go-ai-game/
├── backend/          # Python FastAPI backend
│   ├── app/
│   │   ├── game/    # Go game logic and rules
│   │   ├── ai/      # AI models (random → neural net + MCTS)
│   │   ├── api/     # API routes
│   │   └── db/      # Database models
│   └── main.py
├── frontend/        # React web application
│   └── src/
│       ├── components/  # Go board, game info, etc.
│       └── services/    # API client
└── mobile/         # React Native (future)
```

## Quick Start

### Prerequisites

- Python 3.8+
- Node.js 16+
- npm or yarn

### Backend Setup

1. Navigate to backend directory:
```bash
cd go-ai-game/backend
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the server:
```bash
python main.py
```

Backend will run on `http://localhost:8000`

API docs available at `http://localhost:8000/docs`

### Frontend Setup

1. Navigate to frontend directory:
```bash
cd go-ai-game/frontend
```

2. Install dependencies:
```bash
npm install
```

3. Run development server:
```bash
npm run dev
```

Frontend will run on `http://localhost:3000`

## How to Play

1. Start both backend and frontend servers
2. Open `http://localhost:3000` in your browser
3. Click on the board to place stones (you play as Black)
4. AI will automatically respond (currently random moves)
5. Click "Pass Turn" to pass
6. Game ends after two consecutive passes
7. Click "Save Game" to store the game for replay
8. Click "View Saved Games" to see and replay previous games

## Game Rules

- Standard Go rules with Chinese scoring
- Ko rule enforced
- Suicide moves not allowed
- Area scoring (territory + stones + captures)

## Next Steps: AlphaGo Zero AI

The current AI makes random moves. To implement AlphaGo Zero:

### 1. Neural Network (Policy + Value Network)

The neural network will:
- Take board state as input (9x9x3 tensor: current player stones, opponent stones, current player indicator)
- Output move probabilities (policy head) and position evaluation (value head)
- Use residual CNN architecture

File: `backend/app/ai/neural_net.py`

### 2. Monte Carlo Tree Search (MCTS)

MCTS will:
- Use neural network to guide tree search
- Select promising moves to explore
- Expand tree with network predictions
- Backup values through the tree

File: `backend/app/ai/mcts.py`

### 3. Self-Play Training

Training pipeline will:
- Play games against itself
- Generate training data (state, policy, value)
- Train neural network on self-play data
- Iterate to improve strength

File: `backend/app/ai/training.py`

## Upgrading to AlphaGo Zero

I can help you implement the AlphaGo Zero components. The architecture includes:

1. **Neural Network** - Policy and value heads with shared ResNet backbone
2. **MCTS** - Tree search guided by neural network
3. **Self-Play** - Generate training data by playing against itself
4. **Training Loop** - Continuously improve the model

Would you like me to implement these components next?

## Future Enhancements

- [ ] AlphaGo Zero neural network + MCTS
- [ ] Self-play training pipeline
- [ ] Model versioning and comparison
- [ ] Adjustable board sizes (9x9, 13x13, 19x19)
- [ ] React Native mobile app
- [ ] Deployment to app stores
- [ ] Online multiplayer
- [ ] Game analysis tools
- [ ] Opening book and joseki patterns

## Tech Stack

**Backend:**
- Python 3.8+
- FastAPI - REST API framework
- PyTorch - Deep learning
- SQLAlchemy - Database ORM
- SQLite - Database

**Frontend:**
- React 18
- Vite - Build tool
- Axios - HTTP client
- CSS3 - Styling

**Future:**
- React Native - Mobile apps
- Docker - Containerization
- AWS/GCP - Cloud deployment

## API Endpoints

- `POST /api/game/new` - Create new game
- `GET /api/game/{game_id}` - Get game state
- `POST /api/game/move` - Make a move
- `POST /api/game/pass` - Pass turn
- `GET /api/game/{game_id}/legal-moves` - Get legal moves
- `POST /api/game/{game_id}/save` - Save game
- `GET /api/games` - List saved games
- `GET /api/game/replay/{db_id}` - Get game for replay

## License

MIT

## Contributing

Contributions welcome! Please open an issue or PR.
