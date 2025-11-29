import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from typing import Tuple

class PPONetwork(nn.Module):
    """
    PyTorch Neural Network model for PPO agent.
    It takes grid and vector observations, processes them through separate
    encoders, concatenates the results, and outputs policy logits and a value scalar.
    """
    def __init__(self, grid_channels: int, grid_height: int, grid_width: int,
                 vector_features: int, num_actions: int = 5):
        super().__init__()

        self.grid_channels = grid_channels
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.vector_features = vector_features
        self.num_actions = num_actions

        # CNN encoder for grid input [C, H, W]
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(grid_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # Downsample to reduce dimensionality
        )

        # Calculate the output size of the CNN encoder
        # Simulate a forward pass to get the shape dynamically
        dummy_grid_input = torch.zeros(1, grid_channels, grid_height, grid_width)
        cnn_out_h = self.cnn_encoder(dummy_grid_input).shape[2]
        cnn_out_w = self.cnn_encoder(dummy_grid_input).shape[3]
        self.cnn_output_dim = 64 * cnn_out_h * cnn_out_w

        # MLP encoder for vector input
        self.mlp_encoder = nn.Sequential(
            nn.Linear(vector_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.mlp_output_dim = 64

        # Fully connected layers after concatenation
        self.fc_layers = nn.Sequential(
            nn.Linear(self.cnn_output_dim + self.mlp_output_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.fc_output_dim = 128

        # Policy head
        self.policy_head = nn.Linear(self.fc_output_dim, num_actions)

        # Value head
        self.value_head = nn.Linear(self.fc_output_dim, 1)

    def forward(self, grid_input: torch.Tensor, vector_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.

        Args:
            grid_input (torch.Tensor): Tensor of shape [N, C, H, W] for grid features.
            vector_input (torch.Tensor): Tensor of shape [N, F] for scalar features.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - policy_logits (torch.Tensor): Logits for actions, shape [N, num_actions].
                - value (torch.Tensor): Predicted state value, shape [N, 1].
        """
        # Process grid input
        cnn_out = self.cnn_encoder(grid_input)
        cnn_out = cnn_out.view(-1, self.cnn_output_dim) # Flatten

        # Process vector input
        mlp_out = self.mlp_encoder(vector_input)

        # Concatenate and pass through fully connected layers
        combined_input = torch.cat((cnn_out, mlp_out), dim=-1)
        fc_out = self.fc_layers(combined_input)

        # Get policy logits and value
        policy_logits = self.policy_head(fc_out)
        value = self.value_head(fc_out)

        return policy_logits, value

    def save_checkpoint(self, path: str, optimizer_state_dict=None):
        """
        Saves the model's state dictionary and optionally the optimizer's state dictionary.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'grid_channels': self.grid_channels,
            'grid_height': self.grid_height,
            'grid_width': self.grid_width,
            'vector_features': self.vector_features,
            'num_actions': self.num_actions,
        }
        if optimizer_state_dict is not None:
            checkpoint['optimizer_state_dict'] = optimizer_state_dict
        torch.save(checkpoint, path)

    @classmethod
    def load_checkpoint(cls, path: str, device: torch.device):
        """
        Loads a model from a checkpoint file.
        Returns the model and optionally the optimizer state dict if saved.
        """
        checkpoint = torch.load(path, map_location=device)
        model = cls(
            grid_channels=checkpoint['grid_channels'],
            grid_height=checkpoint['grid_height'],
            grid_width=checkpoint['grid_width'],
            vector_features=checkpoint['vector_features'],
            num_actions=checkpoint['num_actions']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        optimizer_state_dict = checkpoint.get('optimizer_state_dict', None)
        return model, optimizer_state_dict
