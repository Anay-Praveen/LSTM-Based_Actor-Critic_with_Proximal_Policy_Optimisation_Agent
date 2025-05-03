import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from utils import device

class LSTM(nn.Module):
    """LSTM module to process sequential observations."""

    def __init__(self, input_dim, hidden_dim, lstm_layers=5):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lstm_layers = lstm_layers

        # Input preprocessing layer with LayerNorm
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        ).to(device)

        # LSTM for sequential processing
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True # Expects input shape [batch, seq, features]
        ).to(device)

        self._init_weights()

    def _init_weights(self):
        """Initialize LSTM weights using orthogonal initialization for better training stability."""
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, x, hidden=None):
        """Forward pass through LSTM with better shape handling."""
        # Ensure LSTM weights are contiguous
        self.lstm.flatten_parameters()

        # x is expected to be [batch_size, sequence_length, input_dim]
        # Process through input layer
        # Need to apply input_layer to each element in the sequence
        batch_size, sequence_length, _ = x.size()
        x_reshaped = x.view(-1, self.input_dim) # Flatten batch and sequence dims
        features = self.input_layer(x_reshaped) # Apply input layer
        features = features.view(batch_size, sequence_length, self.hidden_dim) # Reshape back to [batch, seq, hidden_dim]


        # LSTM processing
        # If hidden is None, LSTM initializes to zeros
        lstm_out, hidden = self.lstm(features, hidden) # lstm_out is [batch_size, sequence_length, hidden_dim]

        # Return output of last timestep for each item in the batch, and the final hidden state
        return lstm_out[:, -1], hidden # lstm_out[:, -1] is [batch_size, hidden_dim]


class A2CNetwork(nn.Module): # Renamed from PPONetwork as requested
    """Policy and Value networks for A2C algorithm."""

    def __init__(self, input_dim, hidden_dim, action_dim):
        super(A2CNetwork, self).__init__()

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        ).to(device)

        # Value head (critic)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        ).to(device)

        # Policy head (actor)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        ).to(device)

    def forward(self, x):
        """Forward pass returning both value and policy logits."""
        # x is expected to be [batch_size, input_dim] (output from LSTM last timestep)
        features = self.shared(x)
        value = self.value_head(features)
        policy_logits = self.policy_head(features)
        return value, policy_logits, features

    def get_dist(self, x):
        """Get action distribution from features."""
        # x is expected to be [batch_size, input_dim] (output from LSTM last timestep)
        _, logits, _ = self.forward(x)
        return Categorical(logits=logits)

    def get_action(self, x, deterministic=False):
        """Sample action from policy distribution."""
        # x is expected to be [batch_size, input_dim] (output from LSTM last timestep)
        dist = self.get_dist(x)
        if deterministic:
            action = dist.probs.argmax(dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob

    def evaluate_actions(self, x, actions):
        """Evaluate log probs, values and entropy for given states and actions."""
        # x is expected to be [batch_size, input_dim] (output from LSTM last timestep)
        # actions is expected to be [batch_size]
        value, _, _ = self.forward(x) # value is [batch_size, 1]
        dist = self.get_dist(x)
        log_probs = dist.log_prob(actions) # log_probs is [batch_size]
        entropy = dist.entropy() # entropy is [batch_size]
        return log_probs, value, entropy

