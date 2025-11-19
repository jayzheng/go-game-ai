# AlphaGo Zero Training Guide

This guide explains how to train your own AlphaGo Zero style AI for the Go game.

## Overview

The training process follows the AlphaGo Zero methodology:

1. **Self-Play**: The AI plays against itself using Monte Carlo Tree Search (MCTS) guided by a neural network
2. **Training**: The neural network is trained on the game data from self-play
3. **Iteration**: Repeat steps 1-2 to continuously improve

## Quick Start

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Start Training (Quick Test)

For a quick test with minimal resources:

```bash
python train.py --iterations 5 --games 10 --simulations 50 --epochs 5
```

This will:
- Run 5 training iterations
- Play 10 self-play games per iteration
- Use 50 MCTS simulations per move (lower for speed)
- Train for 5 epochs per iteration

### 3. Full Training (Better AI)

For better results (requires more time and compute):

```bash
python train.py --iterations 50 --games 100 --simulations 400 --epochs 20
```

## Training Parameters

### `--iterations` (default: 10)
Number of complete training cycles. More iterations = stronger AI but longer training time.

**Recommendations:**
- Quick test: 5-10
- Decent AI: 20-50
- Strong AI: 100+

### `--games` (default: 100)
Number of self-play games per iteration. More games = more diverse training data.

**Recommendations:**
- Quick test: 10-20
- Normal: 100-200
- Production: 500+

### `--simulations` (default: 100)
MCTS simulations per move. More simulations = better moves but slower.

**Recommendations:**
- Quick test: 50-100
- Normal: 200-400
- Strong play: 800-1600 (like AlphaGo Zero)

### `--epochs` (default: 10)
Training epochs per iteration on collected data.

**Recommendations:**
- Quick test: 5
- Normal: 10-20
- Thorough: 30+

### `--batch-size` (default: 32)
Training batch size. Larger batches are faster but need more memory.

**Recommendations:**
- Limited RAM: 16-32
- Normal: 32-64
- GPU with lots of VRAM: 128-256

### `--board-size` (default: 9)
Size of the Go board.

**Recommendations:**
- Learning/Testing: 9x9 (fastest training)
- Intermediate: 13x13
- Professional: 19x19 (very slow training)

## Training Pipeline Details

### Phase 1: Self-Play

The AI plays complete games against itself:

1. Uses current neural network to evaluate positions
2. Uses MCTS to search for best moves
3. Records game states, move probabilities, and outcomes
4. Stores this data for training

**Output**: Training examples (board position, move probabilities, game outcome)

### Phase 2: Training

The neural network learns from self-play data:

1. **Policy Head**: Learns to predict good moves (supervised by MCTS search results)
2. **Value Head**: Learns to evaluate positions (supervised by game outcomes)

**Output**: Updated neural network weights

### Phase 3: Iteration

The improved network is used for the next round of self-play, creating a continuous improvement loop.

## Monitoring Training

During training, you'll see output like:

```
Epoch 1/10: Loss=2.1543, Policy Loss=1.8234, Value Loss=0.3309
Epoch 2/10: Loss=1.9876, Policy Loss=1.6543, Value Loss=0.3333
...
```

**What to look for:**
- Loss should generally decrease over time
- If loss stops decreasing, the model may have converged
- Very low loss might indicate overfitting

## Testing Your Model

After training, test the model:

```bash
python test_model.py --model models/model_20240101_120000.pt --simulations 200
```

This plays a game between your trained AI (Black) and a random player (White).

## Using Trained Model in Web App

To use your trained model in the web interface:

1. Update `backend/app/api/routes.py` to use `AlphaZeroPlayer` instead of `RandomPlayer`
2. Point it to your trained model file

Example modification in `routes.py`:

```python
from ..ai.alphazero_player import AlphaZeroPlayer

# In create_new_game or similar
ai_player = AlphaZeroPlayer(
    model_path="models/model_20240101_120000.pt",
    num_simulations=200
)
```

## Training Tips

### Start Small
- Begin with 9x9 board
- Use fewer simulations initially
- Test the pipeline works before long training

### Monitor Resources
- **CPU**: 100% usage is normal during self-play
- **Memory**: Watch for out-of-memory errors
- **Disk**: Models and training data can get large

### Incremental Improvement
- Save models regularly
- Test intermediate models
- Compare new models vs old ones

### GPU Acceleration
If you have a CUDA-compatible GPU:

```bash
# PyTorch will automatically use GPU
# Check with:
python -c "import torch; print(torch.cuda.is_available())"
```

GPU can speed up training by 10-50x!

## Advanced Configuration

### Custom Neural Network Architecture

Edit `backend/app/ai/neural_net.py`:

```python
# Change number of residual blocks (more = larger model)
GoNet(board_size=9, num_channels=128, num_res_blocks=10)

# Increase channels for more capacity
GoNet(board_size=9, num_channels=256, num_res_blocks=5)
```

### MCTS Configuration

Edit `backend/app/ai/mcts.py`:

```python
# Adjust exploration constant
MCTS(neural_net, num_simulations=800, c_puct=1.5)
```

Higher `c_puct` = more exploration, lower = more exploitation

## Expected Training Times

**9x9 Board (CPU - Apple M1/M2 or equivalent)**
- Quick test (10 iterations, 10 games, 50 sims): ~30 minutes
- Normal (20 iterations, 100 games, 200 sims): ~8-10 hours
- Strong (50 iterations, 200 games, 400 sims): ~48-72 hours

**9x9 Board (GPU - NVIDIA RTX 3080 or equivalent)**
- Quick test: ~5-10 minutes
- Normal: ~1-2 hours
- Strong: ~6-12 hours

**19x19 Board**
- Multiply above times by approximately 4-5x
- Requires significantly more training iterations for good play

## Troubleshooting

### "Out of Memory" Error
- Reduce `--batch-size`
- Reduce `--games`
- Use smaller neural network (fewer channels/blocks)

### Training is Too Slow
- Reduce `--simulations`
- Reduce `--games`
- Use GPU if available
- Start with 9x9 board

### Model Not Improving
- Increase `--iterations`
- Increase `--games` for more diverse data
- Try different learning rate in `training.py`
- Check if loss is decreasing

### Model Plays Poorly
- Train for more iterations
- Increase MCTS simulations
- Collect more self-play data
- Ensure training loss is decreasing

## Next Steps

After training:

1. Test your model against random play
2. Compare different model versions
3. Adjust hyperparameters and retrain
4. Integrate best model into web app
5. Share games and analyze play patterns

## References

- [AlphaGo Zero Paper](https://www.nature.com/articles/nature24270)
- [Mastering the Game of Go without Human Knowledge](https://discovery.ucl.ac.uk/id/eprint/10045895/1/agz_unformatted_nature.pdf)

Happy training!
