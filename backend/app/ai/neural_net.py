"""
Neural Network for AlphaGo Zero style AI
Combines policy and value heads with residual CNN backbone
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class ResidualBlock(nn.Module):
    """Residual CNN block"""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class GoNet(nn.Module):
    """
    AlphaGo Zero style neural network

    Architecture:
    - Input: Board state (board_size x board_size x 3)
      - Channel 0: Current player stones
      - Channel 1: Opponent stones
      - Channel 2: Current player indicator (all 1s or 0s)
    - Convolutional layers with residual blocks
    - Policy head: Outputs move probabilities
    - Value head: Outputs position evaluation
    """

    def __init__(self, board_size: int = 9, num_channels: int = 128, num_res_blocks: int = 5):
        super().__init__()
        self.board_size = board_size
        self.num_channels = num_channels

        # Initial convolutional layer
        self.conv_input = nn.Conv2d(3, num_channels, kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(num_channels)

        # Residual tower
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_channels) for _ in range(num_res_blocks)
        ])

        # Policy head
        self.policy_conv = nn.Conv2d(num_channels, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, board_size * board_size + 1)

        # Value head
        self.value_conv = nn.Conv2d(num_channels, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            x: Input tensor of shape (batch, 3, board_size, board_size)

        Returns:
            policy: Move probabilities (batch, board_size*board_size + 1)
            value: Position evaluation (batch, 1)
        """
        # Initial conv
        out = F.relu(self.bn_input(self.conv_input(x)))

        # Residual tower
        for block in self.res_blocks:
            out = block(out)

        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(out)))
        policy = policy.view(-1, 2 * self.board_size * self.board_size)
        policy = self.policy_fc(policy)
        policy = F.log_softmax(policy, dim=1)

        # Value head
        value = F.relu(self.value_bn(self.value_conv(out)))
        value = value.view(-1, self.board_size * self.board_size)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return policy, value

    def predict(self, board_state: np.ndarray, current_player: int) -> Tuple[np.ndarray, float]:
        """
        Predict policy and value for a single board state

        Args:
            board_state: Board as numpy array (board_size, board_size)
            current_player: Current player (1 for black, 2 for white)

        Returns:
            policy: Move probabilities
            value: Position evaluation (-1 to 1)
        """
        self.eval()
        with torch.no_grad():
            # Convert board to input tensor
            input_tensor = self.board_to_tensor(board_state, current_player)
            input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension

            # Forward pass
            policy_log, value = self.forward(input_tensor)

            # Convert to numpy
            policy = torch.exp(policy_log).cpu().numpy()[0]
            value = value.cpu().numpy()[0][0]

        return policy, value

    def board_to_tensor(self, board_state: np.ndarray, current_player: int) -> torch.Tensor:
        """
        Convert board state to input tensor

        Args:
            board_state: Board as numpy array where 0=empty, 1=black, 2=white
            current_player: Current player (1=black, 2=white)

        Returns:
            Input tensor of shape (3, board_size, board_size)
        """
        board_size = board_state.shape[0]
        tensor = np.zeros((3, board_size, board_size), dtype=np.float32)

        # Channel 0: Current player stones
        tensor[0] = (board_state == current_player).astype(np.float32)

        # Channel 1: Opponent stones
        opponent = 2 if current_player == 1 else 1
        tensor[1] = (board_state == opponent).astype(np.float32)

        # Channel 2: Current player indicator
        tensor[2] = np.full((board_size, board_size), current_player - 1, dtype=np.float32)

        return torch.from_numpy(tensor)

    def save_checkpoint(self, path: str, optimizer=None, epoch: int = 0):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'board_size': self.board_size,
            'num_channels': self.num_channels,
        }
        if optimizer:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

    @classmethod
    def load_checkpoint(cls, path: str, optimizer=None):
        """Load model checkpoint"""
        checkpoint = torch.load(path)

        model = cls(
            board_size=checkpoint['board_size'],
            num_channels=checkpoint['num_channels']
        )
        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        print(f"Checkpoint loaded from {path}, epoch {checkpoint['epoch']}")
        return model, checkpoint['epoch']
