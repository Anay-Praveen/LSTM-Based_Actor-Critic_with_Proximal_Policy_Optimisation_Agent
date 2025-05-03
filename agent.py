import gymnasium as gym
import nasim
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import time
import os
import datetime
import json
import csv

from typing import Dict, List, Tuple, Optional, Union, Any
from torch.distributions import Categorical # Import Categorical here


# Import modules
from utils import device, logger, NumpyEncoder, PPOTransition # Import necessary items from utils, including PPOTransition
from networks import LSTM, A2CNetwork # Import renamed A2CNetwork and LSTM
from memory import A2CMemory # Import A2CMemory
from plot import plot_rolling_averages # Import the plotting function from the new file

class NASIMOffensiveAgent:
    """Optimized LSTM-A2C Agent for Network Penetration Testing.""" # Updated description

    def __init__(self, env, hidden_size=128, lstm_layers=5, learning_rate=0.0003,
                 gamma=0.99, gae_lambda=0.95, clip_ratio=0.2,
                 value_coef=0.5, entropy_coef=0.01, max_grad_norm=0.5,
                 ppo_epochs=4, ppo_batch_size=64, reward_clip=[-20, 20],
                 progress_advantage_alpha=1.0): # Added alpha hyperparameter

        # Environment setup
        self.env = env
        self.obs_dim = self._get_obs_dim(env)
        self.action_dim = env.action_space.n

        # Try to get target hosts from environment
        try:
            if hasattr(env, 'scenario') and hasattr(env.scenario, 'target_hosts'):
                self.target_hosts = set(env.scenario.target_hosts)
            else:
                self.target_hosts = None
        except:
            self.target_hosts = None  # Will rely on other success detection methods

        # Neural network components (Using A2CNetwork)
        self.lstm = LSTM(self.obs_dim, hidden_size, lstm_layers).to(device)
        self.a2c = A2CNetwork(hidden_size, hidden_size, self.action_dim).to(device) # Using A2CNetwork

        # Optimizer
        self.optimizer = optim.Adam(
            list(self.lstm.parameters()) + list(self.a2c.parameters()), # Updated parameters
            lr=learning_rate
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau( # Corrected typo: ReduceLROnPlateau
            self.optimizer, mode='max', factor=0.5, patience=20
        )

        # PPO Hyperparameters (Keeping original PPO names for these parameters)
        self.clip_ratio = clip_ratio # Keeping as per original, though less standard for A2C
        self.gamma = gamma
        self.gae_lambda = gae_lambda # Keeping GAE as per original
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_entropy_coef = 0.1
        self.min_entropy_coef = 0.005
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs # Keeping original name
        self.ppo_batch_size = ppo_batch_size # Keeping original name
        self.reward_clip = reward_clip
        self.progress_advantage_alpha = progress_advantage_alpha # Store alpha

        # Reward history for adaptive clipping
        self.reward_history = deque(maxlen=1000)

        # A2C Memory (Using A2CMemory name)
        self.memory = A2CMemory(batch_size=ppo_batch_size) # Using A2CMemory

        # LSTM State Management
        self.sequence_length = 10 # Keeps the sequence length consistent
        self.obs_buffer = deque(maxlen=self.sequence_length)
        self.hidden = None # LSTM hidden state

        # Training statistics and metrics
        self.episode_metrics = {
            'rewards': [],
            'lengths': [],
            'successes': [],
            'entropies': []
        }
        self.episode_count = 0
        self.successful_episodes = 0

        # Curriculum tracking
        self.current_scenario = None
        self.scenario_history = []
        self.scenario_performances = {}

        # Environment-specific hyperparameter settings
        # Tuned alpha values based on potential scenario complexity
        self.scenario_settings = {
            'tiny': {'entropy_coef': 0.01, 'learning_rate': 0.0003, 'clip_ratio': 0.2, 'progress_advantage_alpha': 0.5}, # Lower alpha for simpler env
            'tiny-hard': {'entropy_coef': 0.03, 'learning_rate': 0.0002, 'clip_ratio': 0.2, 'progress_advantage_alpha': 0.8},
            'tiny-small': {'entropy_coef': 0.04, 'learning_rate': 0.0001, 'clip_ratio': 0.15, 'progress_advantage_alpha': 1.0},
            'small': {'entropy_coef': 0.05, 'learning_rate': 0.0001, 'clip_ratio': 0.15, 'progress_advantage_alpha': 1.2}, # Higher alpha for more complex env
            'small-honeypot': {'entropy_coef': 0.05, 'learning_rate': 0.0001, 'clip_ratio': 0.15, 'progress_advantage_alpha': 1.2},
            'small-linear': {'entropy_coef': 0.04, 'learning_rate': 0.0001, 'clip_ratio': 0.15, 'progress_advantage_alpha': 1.0},
            'medium': {'entropy_coef': 0.05, 'learning_rate': 0.00008, 'clip_ratio': 0.1, 'progress_advantage_alpha': 1.5}, # Higher alpha for medium env
            'medium-single-site': {'entropy_coef': 0.05, 'learning_rate': 0.00008, 'clip_ratio': 0.1, 'progress_advantage_alpha': 1.3},
            'medium-multi-site': {'entropy_coef': 0.06, 'learning_rate': 0.00008, 'clip_ratio': 0.1, 'progress_advantage_alpha': 1.6}, # Highest alpha for multi-site
        }

        # Logging setup
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.metrics_dir = f"metrics_Plot"
        os.makedirs(self.metrics_dir, exist_ok=True)
        self.episode_log_file = f"{self.metrics_dir}/episode_logs.csv"
        with open(self.episode_log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'Scenario', 'Episode', 'Reward', 'Steps', 'Success',
                             'Entropy', 'EntropyCoef', 'MostCommonAction', 'ActionFrequency'])

        # Log model size and initial GPU memory usage
        self._log_model_info()

    def _log_model_info(self):
        """Log model size and initial GPU memory utilization."""
        total_params = sum(p.numel() for p in self.lstm.parameters() if p.requires_grad)
        total_params += sum(p.numel() for p in self.a2c.parameters() if p.requires_grad) # Using A2CNetwork
        logger.info(f"Model initialized with {total_params:,} trainable parameters")
        if torch.cuda.is_available():
            logger.info(f"Initial GPU memory: {torch.cuda.memory_allocated() / 1024 ** 2:.1f}MB / "
                        f"{torch.cuda.memory_reserved() / 1.024 ** 2:.1f}MB") # Corrected typo


    def _get_obs_dim(self, env):
        """Compute observation dimension from environment."""
        if hasattr(env.observation_space, 'spaces'):
            return sum(np.prod(space.shape) for space in env.observation_space.spaces.values())
        return np.prod(env.observation_space.shape)

    def _apply_scenario_settings(self, scenario_name):
        """Apply environment-specific hyperparameters."""
        if scenario_name in self.scenario_settings:
            settings = self.scenario_settings[scenario_name]

            # Apply entropy coefficient
            if 'entropy_coef' in settings:
                self.entropy_coef = settings['entropy_coef']
                logger.info(f"Setting entropy coefficient to {self.entropy_coef} for {scenario_name}")

            # Apply learning rate
            if 'learning_rate' in settings:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = settings['learning_rate']
                logger.info(f"Setting learning rate to {settings['learning_rate']} for {scenario_name}")

            # Apply clip ratio
            if 'clip_ratio' in settings:
                self.clip_ratio = settings['clip_ratio']
                logger.info(f"Setting clip ratio to {self.clip_ratio} for {scenario_name}")

            # Apply progress advantage alpha
            if 'progress_advantage_alpha' in settings:
                self.progress_advantage_alpha = settings['progress_advantage_alpha']
                logger.info(f"Setting progress advantage alpha to {self.progress_advantage_alpha} for {scenario_name}")


            # Reset reward clipping for new scenario
            self.reward_clip = [-20, 20]
        else:
            # Default settings if scenario not explicitly configured
            self.entropy_coef = 0.01
            self.clip_ratio = 0.2
            self.progress_advantage_alpha = 1.0 # Default alpha

    def _update_entropy_coefficient(self):
        """Dynamically adjust entropy coefficient based on performance."""
        if len(self.episode_metrics['successes']) < 10:
            return

        # Get recent success rate
        recent_successes = self.episode_metrics['successes'][-20:]
        recent_success_rate = sum(recent_successes) / len(recent_successes)
        recent_entropies = self.episode_metrics['entropies'][-20:]
        avg_entropy = sum(recent_entropies) / max(1, len(recent_entropies))

        # Set target entropy based on performance
        if recent_success_rate < 0.1:
            # Very low success rate - increase entropy substantially
            target_entropy = min(self.max_entropy_coef, self.entropy_coef * 2.0)
        elif recent_success_rate < 0.3:
            # Below 30% success - increase entropy moderately
            target_entropy = min(self.max_entropy_coef, self.entropy_coef * 1.2)
        elif recent_success_rate > 0.8 and avg_entropy < 0.5:
            # High success rate and low entropy - reduce entropy
            target_entropy = max(self.min_entropy_coef, self.entropy_coef * 0.9)
        else:
            # Otherwise maintain current entropy
            return

        # Smooth update to entropy coefficient
        self.entropy_coef = 0.95 * self.entropy_coef + 0.05 * target_entropy
        self.entropy_coef = max(self.min_entropy_coef, min(self.max_entropy_coef, self.entropy_coef))

        logger.info(
            f"Adjusted entropy coefficient to {self.entropy_coef:.6f} (success rate: {recent_success_rate:.2f})")

    def _update_reward_clipping(self):
        """Dynamically adjust reward clipping based on recent rewards."""
        if len(self.reward_history) < 50:
            return

        # Use percentiles to avoid outliers
        min_reward = np.percentile(list(self.reward_history), 10)
        max_reward = np.percentile(list(self.reward_history), 90)

        # Ensure reasonable bounds
        min_reward = max(-200, min_reward)
        max_reward = min(1000, max_reward)

        # Smoothly update reward clipping
        self.reward_clip[0] = 0.9 * self.reward_clip[0] + 0.1 * min_reward
        self.reward_clip[1] = 0.9 * self.reward_clip[1] + 0.1 * max_reward

    def preprocess_observation(self, observation):
        """Convert observation to tensor and add to sequence buffer."""
        # Handle different observation types and flatten
        flat_obs = None
        try:
            if isinstance(observation, dict):
                # Dictionary observations: flatten values
                flat_parts = []
                for k, v in observation.items():
                    if isinstance(v, np.ndarray):
                        flat_parts.append(v.flatten())
                    elif isinstance(v, (int, float, np.number)):
                        flat_parts.append(np.array([float(v)]))
                    elif isinstance(v, (list, tuple)):
                         flat_parts.append(np.array(v, dtype=float).flatten())
                    else:
                        logger.warning(f"Unsupported observation dict value type for key '{k}': {type(v)}. Attempting conversion.")
                        try:
                            flat_parts.append(np.array([float(v)]).flatten())
                        except:
                            logger.warning(f"Failed to convert unsupported type for key '{k}'. Using zero vector.")
                            flat_parts.append(np.zeros(1)) # Use a zero vector as fallback


                if flat_parts:
                    # Ensure all parts are float32 and concatenate
                    flat_parts = [part.astype(np.float32) for part in flat_parts if part.size > 0] # Filter out empty parts
                    if flat_parts:
                        flat_obs = np.concatenate(flat_parts)
                    else:
                        flat_obs = np.zeros(self.obs_dim, dtype=np.float32) # Fallback if all parts were empty

            elif isinstance(observation, (tuple, list)):
                # Tuple or List observations: flatten elements
                flat_parts = []
                for item in observation:
                    if isinstance(item, (int, float, np.number)):
                        flat_parts.append(np.array([float(item)]))
                    elif isinstance(item, np.ndarray):
                        flat_parts.append(item.flatten())
                    elif isinstance(item, (list, tuple)):
                        flat_parts.append(np.array(item, dtype=float).flatten())
                    else:
                        logger.warning(f"Unsupported observation list/tuple element type: {type(item)}. Attempting conversion.")
                        try:
                            flat_parts.append(np.array([float(item)]).flatten())
                        except:
                            logger.warning(f"Failed to convert unsupported type. Using zero vector.")
                            flat_parts.append(np.zeros(1)) # Use a zero vector as fallback


                if flat_parts:
                    # Ensure all parts are float32 and concatenate
                    flat_parts = [part.astype(np.float32) for part in flat_parts if part.size > 0] # Filter out empty parts
                    if flat_parts:
                        flat_obs = np.concatenate(flat_parts)
                    else:
                        flat_obs = np.zeros(self.obs_dim, dtype=np.float32) # Fallback if all parts were empty

            elif isinstance(observation, np.ndarray):
                flat_obs = observation.flatten().astype(np.float32)
            else:
                # Try direct conversion for primitive types
                try:
                    flat_obs = np.array([float(observation)], dtype=np.float32).flatten()
                except:
                    logger.warning(f"Unsupported observation type: {type(observation)}. Using zero vector.")
                    flat_obs = np.zeros(self.obs_dim, dtype=np.float32) # Fallback for unhandled types

        except Exception as e:
            # Catch any unexpected errors during processing
            logger.error(f"Critical error during observation preprocessing: {e}. Using zero vector.")
            flat_obs = np.zeros(self.obs_dim, dtype=np.float32)


        # Ensure observation matches expected dimension
        if flat_obs is None or flat_obs.size == 0:
             logger.warning("Observation preprocessing resulted in empty array. Using zero vector.")
             flat_obs = np.zeros(self.obs_dim, dtype=np.float32)


        if len(flat_obs) != self.obs_dim:

            # The padding/truncating handles the mismatch for inference.
            # logger.warning(f"Observation dimension mismatch: Expected {self.obs_dim}, got {len(flat_obs)}. Padding or truncating.")
            if len(flat_obs) < self.obs_dim:
                # Pad with zeros
                padded_obs = np.pad(flat_obs, (0, self.obs_dim - len(flat_obs)), 'constant')
                flat_obs = padded_obs
            else:
                # Truncate
                flat_obs = flat_obs[:self.obs_dim]


        # Normalize and convert to tensor
        # Added a small epsilon to avoid division by zero if max value is 0
        max_val = np.max(flat_obs) if flat_obs.size > 0 else 100.0
        flat_obs = np.clip(flat_obs, 0, max_val) / (max_val + 1e-8) * 100.0 # Scale to approx 0-100 range
        flat_obs = np.clip(flat_obs, 0, 100) / 100.0 # Normalize to 0-1 range

        obs_tensor = torch.tensor(flat_obs, dtype=torch.float32, device=device)

        # Add to observation buffer
        self.obs_buffer.append(obs_tensor)

        # Pad buffer if not full
        if len(self.obs_buffer) < self.sequence_length:
            padding = [torch.zeros_like(obs_tensor) for _ in range(self.sequence_length - len(self.obs_buffer))]
            padded_buffer = padding + list(self.obs_buffer)
        else:
            padded_buffer = list(self.obs_buffer)

        # Create sequence tensor [1, sequence_length, obs_dim]
        sequence = torch.stack(padded_buffer).unsqueeze(0)

        return sequence

    def select_action(self, observation, action_mask=None, deterministic=False):
        """Select action based on current policy, with optional action masking."""
        with torch.no_grad():
            # Process observation through LSTM
            state_seq = self.preprocess_observation(observation)

            # Ensure LSTM weights are contiguous
            self.lstm.lstm.flatten_parameters()

            # Get LSTM features and update hidden state
            # pass the current hidden state and get the next one
            # state_seq is [1, sequence_length, obs_dim]
            lstm_features, self.hidden = self.lstm(state_seq, self.hidden) # lstm_features is [1, hidden_dim]

            # Get value and policy logits from the A2C network
            value, policy_logits, _ = self.a2c(lstm_features) # Using A2CNetwork

            # Apply action masking if mask is provided
            if action_mask is not None:
                # Convert mask to tensor and move to device
                mask_tensor = torch.tensor(action_mask, dtype=torch.bool, device=device).unsqueeze(0) # Add batch dim

                # Set logits of invalid actions to a very low value (negative infinity)
                # This ensures their probability becomes zero after softmax
                policy_logits[~mask_tensor] = float('-inf')

            # Get action distribution from potentially masked logits
            dist = Categorical(logits=policy_logits)

            # Sample action
            if deterministic:
                action = dist.probs.argmax(dim=-1)
            else:
                action = dist.sample()

            # Calculate log probability of the selected action
            log_prob = dist.log_prob(action)

            return action.item(), log_prob.item(), value.item()

    def clip_reward(self, reward):
        """Clip reward to handle extreme values."""
        return max(self.reward_clip[0], min(self.reward_clip[1], reward))

    def update_policy(self):
        """Update policy using PPO-like update with specialized advantage."""
        # Skip if no data
        if len(self.memory.states) == 0:
            return {'policy_loss': 0, 'value_loss': 0, 'entropy': 0}

        # Track losses
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        update_count = 0


        # states_batch is [batch_size, sequence_length, obs_dim]
        for states_batch, actions_batch, old_log_probs_batch, returns_batch, advantages_batch in self.memory.get_batches():

            # Reset gradients
            self.optimizer.zero_grad()

            # Process states batch through LSTM
            # Pass the entire batch to the LSTM. LSTM will handle the sequence.
            # Initialize hidden state to zeros for each batch (TBPTT assumption)
            # lstm_features_batch will be [batch_size, hidden_dim]
            self.lstm.lstm.flatten_parameters() # Ensure contiguous before forward pass
            lstm_features_batch, _ = self.lstm(states_batch, None) # Pass None for initial hidden state

            # Evaluate actions using the A2C network
            # lstm_features_batch is [batch_size, hidden_dim]
            # actions_batch is [batch_size]
            # Note: Action masking is applied during action selection (inference), not typically during policy evaluation for training.
            # The log_probs and entropy are calculated based on the distribution *without* masking during the update.
            # This is standard for on-policy methods where the data is collected using the *current* policy, which *did* use masking.
            new_log_probs, values, entropy = self.a2c.evaluate_actions(lstm_features_batch, actions_batch) # Using A2CNetwork


            # A2C Policy Loss (using clipped objective from PPO)
            ratio = torch.exp(new_log_probs - old_log_probs_batch)
            surr1 = ratio * advantages_batch
            surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages_batch
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value function loss (using clipped value loss from PPO)
            # Ensure shapes match: values [batch_size, 1], returns_batch [batch_size]
            value_pred_clipped = values + torch.clamp(
                values.squeeze(-1) - returns_batch, # Ensure shapes are compatible for clamp
                -self.clip_ratio, self.clip_ratio
            ).unsqueeze(-1) # Add dimension back for MSE loss

            # Use .squeeze(-1) to remove the last dimension for F.mse_loss
            value_loss = 0.5 * torch.max(
                F.mse_loss(values.squeeze(-1), returns_batch), # Ensure shapes are compatible
                F.mse_loss(value_pred_clipped.squeeze(-1), returns_batch) # Ensure shapes are compatible
            )

            # Entropy bonus for exploration
            entropy_loss = -entropy.mean()

            # Total loss
            loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

            # Backpropagation
            loss.backward()

            # Gradient clipping
            nn.utils.clip_grad_norm_(
                list(self.lstm.parameters()) + list(self.a2c.parameters()), # Updated parameters
                self.max_grad_norm
            )

            # Update parameters
            self.optimizer.step()

            # Track losses
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.mean().item()
            update_count += 1

        # Update entropy coefficient based on performance
        if len(self.episode_metrics['successes']) >= 10:
            self._update_entropy_coefficient()

        # Clear memory buffer
        self.memory.clear()

        # Return average losses
        if update_count > 0:
            return {
                'policy_loss': total_policy_loss / update_count,
                'value_loss': total_value_loss / update_count,
                'entropy': total_entropy / update_count
            }
        return {'policy_loss': 0, 'value_loss': 0, 'entropy': 0}



    def check_success(self, info):
        """Determine if the episode was successful based on environment info."""
        # Different ways to check success in NASim environment
        if info is None:
            return False

        # Check for success flag in info
        if 'success' in info:
            return info['success']

        # Check for goal_reached flag
        if 'goal_reached' in info:
            return info['goal_reached']

        # Check if any target hosts were compromised
        if self.target_hosts and 'compromised_hosts' in info:
            compromised = set(info.get('compromised_hosts', [])) # Use get with default for safety
            return bool(compromised.intersection(self.target_hosts))

        # Check for general compromised flag
        if 'compromised' in info and info['compromised']:
            return True

        # Fallback if no other indicators available
        return False

    def train_on_scenario(self, scenario_name, episodes_to_train):
        """Train the agent on a specific scenario for a fixed number of episodes."""
        # Initialize scenario
        logger.info(f"\n{'=' * 50}")
        logger.info(f"Starting training on scenario: {scenario_name}")

        # Update current scenario
        self.current_scenario = scenario_name

        # Create environment for this scenario
        try:
            # Convert scenario name to NASim format
            scenario_name_formatted = "".join(word.capitalize() for word in scenario_name.split("-"))
            new_env = gym.make(f"nasim:{scenario_name_formatted}PO-v0")

            # Update environment
            self.env = new_env

            # Try to get target hosts for this environment
            try:
                if hasattr(new_env, 'scenario') and hasattr(new_env.scenario, 'target_hosts'):
                    self.target_hosts = set(new_env.scenario.target_hosts)
                else:
                    self.target_hosts = None
            except:
                self.target_hosts = None

            # Apply scenario-specific settings
            self._apply_scenario_settings(scenario_name)

            # Get new observation and action dimensions
            new_obs_dim = self._get_obs_dim(new_env)
            new_action_dim = new_env.action_space.n

            # Log environment information
            logger.info(f"Scenario: {scenario_name}")
            logger.info(f"Observation Space: {new_env.observation_space}")
            logger.info(f"Observation Dim: {new_obs_dim}")
            logger.info(f"Action Space: {new_env.action_space}")
            logger.info(f"Action Dim: {new_action_dim}")
            if self.target_hosts:
                logger.info(f"Target Hosts: {self.target_hosts}")

            # Handle potential changes in observation or action space dimensions
            if new_obs_dim != self.obs_dim or new_action_dim != self.action_dim:
                logger.info(f"Environment dimensions changed. Transferring knowledge...")

                # Store old optimizer state before creating new networks
                old_optimizer_state_dict = self.optimizer.state_dict()

                # Save old networks and dimensions
                old_lstm = self.lstm
                old_a2c = self.a2c # Using A2CNetwork
                old_obs_dim = self.obs_dim
                old_action_dim = self.action_dim

                # Update dimensions
                self.obs_dim = new_obs_dim
                self.action_dim = new_action_dim

                # Create new networks with potentially different input/output dims but same hidden size/layers
                self.lstm = LSTM(self.obs_dim, self.lstm.hidden_dim, self.lstm.lstm_layers).to(device)
                self.a2c = A2CNetwork(self.lstm.hidden_dim, self.lstm.hidden_dim, self.action_dim).to(device) # Using A2CNetwork

                # Transfer compatible weights
                with torch.no_grad():
                    # Copy LSTM weights
                    for i in range(min(self.lstm.lstm_layers, old_lstm.lstm_layers)):
                         self.lstm.lstm.__getattr__(f'weight_hh_l{i}').copy_(old_lstm.lstm.__getattr__(f'weight_hh_l{i}'))
                         self.lstm.lstm.__getattr__(f'bias_hh_l{i}').copy_(old_lstm.lstm.__getattr__(f'bias_hh_l{i}'))
                         self.lstm.lstm.__getattr__(f'weight_ih_l{i}').copy_(old_lstm.lstm.__getattr__(f'weight_ih_l{i}'))
                         self.lstm.lstm.__getattr__(f'bias_ih_l{i}').copy_(old_lstm.lstm.__getattr__(f'bias_ih_l{i}'))

                    # Transfer input layer weights for common features
                    min_features = min(old_obs_dim, new_obs_dim)
                    if min_features > 0:
                        self.lstm.input_layer[0].weight.data[:, :min_features].copy_(
                            old_lstm.input_layer[0].weight.data[:, :min_features]
                        )
                        if hasattr(self.lstm.input_layer[0], 'bias') and hasattr(old_lstm.input_layer[0], 'bias'):
                            self.lstm.input_layer[0].bias.data.copy_(old_lstm.input_layer[0].bias.data)

                    # Copy A2C shared layers and value head (independent of action space)
                    self.a2c.shared.load_state_dict(old_a2c.shared.state_dict()) # Using A2CNetwork
                    self.a2c.value_head.load_state_dict(old_a2c.value_head.state_dict()) # Using A2CNetwork

                    # For policy head, copy weights for common actions
                    min_actions = min(old_action_dim, new_action_dim)
                    if min_actions > 0:
                        self.a2c.policy_head[-1].weight.data[:min_actions, :].copy_(
                            old_a2c.policy_head[-1].weight.data[:min_actions, :]
                        ) # Using A2CNetwork
                        self.a2c.policy_head[-1].bias.data[:min_actions].copy_(
                            old_a2c.policy_head[-1].bias.data[:min_actions]
                        ) # Using A2CNetwork


                # Create a new optimizer instance with the current parameters
                # This is necessary because the model parameters have changed
                new_optimizer = optim.Adam(
                    list(self.lstm.parameters()) + list(self.a2c.parameters()),
                    lr=self.optimizer.param_groups[0]['lr'] # Use the current learning rate
                )

                # does NOT load the old optimizer state when dimensions change
                # because the state tensors are tied to the old parameter sizes.
                # The new optimizer starts with a fresh state but retains the learning rate.

                self.optimizer = new_optimizer

                logger.info(
                    f"Knowledge transferred from {old_obs_dim}→{new_obs_dim} obs dims and {old_action_dim}→{new_action_dim} action dims")

                # Reset buffers for new dimensions
                self.obs_buffer.clear()
                self.hidden = None # Reset LSTM hidden state

        except Exception as e:
            logger.error(f"Failed to create environment for scenario {scenario_name}: {e}")
            return {
                'scenario': scenario_name,
                'completed': False,
                'error': str(e),
                'episodes': 0,
                'avg_reward': 0.0,
                'successful_episodes': 0
            }

        # Initialize training variables
        episodes_completed = 0
        episode_rewards = []
        best_reward = -float('inf')

        # Create scenario directory for checkpoints
        exp_dir = f"scenario_checkpoints/{scenario_name}"
        os.makedirs(exp_dir, exist_ok=True)

        # Reset stats for this scenario
        # These are scenario-specific metrics and should be reset per scenario.
        self.successful_episodes = 0
        self.episode_metrics['rewards'] = []
        self.episode_metrics['lengths'] = []
        self.episode_metrics['successes'] = []
        self.episode_metrics['entropies'] = []
        self.reward_history.clear() # Clear reward history for new clipping range

        scenario_start_time = time.time()

        logger.info(f"Training on {scenario_name} for {episodes_to_train} episodes")

        # Set environment-specific step limit based on scenario
        # Keeping original step limits
        if 'tiny' in scenario_name or 'small' in scenario_name or 'huge' in scenario_name:
            max_steps_per_episode = 1000
        elif 'medium' in scenario_name:
            max_steps_per_episode = 2000
        else:
            max_steps_per_episode = 1000  # Default

        logger.info(f"Setting max steps per episode to {max_steps_per_episode}")

        # Training loop for fixed number of episodes
        while episodes_completed < episodes_to_train:
            # Reset environment and agent state
            obs, info = self.env.reset() # Gymnasium returns obs, info
            self.obs_buffer.clear()
            self.hidden = None # Reset LSTM hidden state
            self.memory._initial_info = info # Store initial info for progress calculation
            self.memory.clear() # Clear memory at the start of each episode for on-policy collection

            # Get the action mask for the initial state
            current_action_mask = info.get('action_mask', None)

            # Reset episode-specific tracking
            compromised_hosts = set()

            # Episode tracking
            done = False
            episode_reward = 0
            episode_steps = 0
            actions_taken = []
            reached_max_steps = False
            is_success = False

            # Final info from environment
            final_info = {}

            # Run until episode terminates or reaches max steps
            while not done and episode_steps < max_steps_per_episode:
                # Select action, passing the current action mask
                action, log_prob, value = self.select_action(obs, action_mask=current_action_mask)
                actions_taken.append(action)

                # Step environment
                try:
                    step_result = self.env.step(action)
                    # Handle both new and old Gym API formats
                    if len(step_result) == 5:  # New Gym API format (obs, reward, terminated, truncated, info)
                        next_obs, reward, terminated, truncated, info = step_result
                        done = terminated or truncated
                    else:  # Old Gym API format (obs, reward, done, info)
                        next_obs, reward, done, info = step_result

                    # Update the action mask for the *next* state
                    current_action_mask = info.get('action_mask', None)


                    # Keep track of final info when episode is about to end
                    if done or episode_steps == max_steps_per_episode - 1:
                         # Make a copy of info as it might be modified later
                        final_info = info.copy() if info is not None else {}


                    # Update compromised hosts tracking from info
                    if info and 'compromised_hosts' in info:
                        if isinstance(info['compromised_hosts'], (list, tuple, set)):
                            compromised_hosts.update(info['compromised_hosts'])
                        elif info['compromised_hosts'] is not None: # Handle non-iterable but not None
                             compromised_hosts.add(info['compromised_hosts'])


                except Exception as e:
                    logger.error(f"Environment step error: {e}")
                    # Terminate episode on error
                    done = True
                    next_obs = obs # Stay in current state or handle appropriately
                    reward = 0 # Assign a penalty or 0 reward on error
                    info = {} # Clear info or add error info
                    current_action_mask = None # Reset mask on error


                # Track raw reward for metrics
                raw_reward = reward
                episode_reward += raw_reward
                episode_steps += 1

                # Clip reward for training stability
                clipped_reward = self.clip_reward(reward)

                # Track rewards for adaptive clipping
                self.reward_history.append(raw_reward)

                # Store transition in memory, including the info dictionary
                current_state_seq = self.preprocess_observation(obs) # Preprocess current state
                self.memory.add(
                    state=current_state_seq, # State before the action
                    action=action,
                    reward=clipped_reward,
                    value=value, # Value of the state *before* the action
                    log_prob=log_prob,
                    mask=1.0 - float(done),  # 0 if done, 1 if not done
                    info=info # Pass the info dictionary
                )

                # Move to next state
                obs = next_obs

            # Check if episode ended due to step limit
            if episode_steps >= max_steps_per_episode:
                reached_max_steps = True

            # Determine if episode was successful (target host compromised)
            # Use the final_info captured just before the episode ended
            is_success = self.check_success(final_info)

            # If we have target hosts defined, check if any were compromised
            # This check uses the accumulated compromised_hosts during the episode
            if not is_success and self.target_hosts and compromised_hosts:
                is_success = bool(compromised_hosts.intersection(self.target_hosts))

            # End of episode processing
            # Compute GAE returns if there's data in memory
            if len(self.memory.states) > 0:
                # Get value estimate for final state if not terminal
                with torch.no_grad():
                    if not done:
                        # Need to preprocess the final observation to get its value
                        # Use the hidden state from the end of the episode
                        final_obs_seq = self.preprocess_observation(obs)
                        # Ensure LSTM weights are contiguous before forward pass
                        self.lstm.lstm.flatten_parameters()
                        final_lstm_features, _ = self.lstm(final_obs_seq, self.hidden) # Use the final hidden state
                        last_value, _, _ = self.a2c(final_lstm_features) # Using A2CNetwork
                        last_value = last_value.item()
                    else:
                        last_value = 0

                    # Compute returns and advantages using the specialized function
                    self.memory.compute_returns_and_advantages(
                        last_value,
                        gamma=self.gamma,
                        gae_lambda=self.gae_lambda,
                        alpha=self.progress_advantage_alpha # Pass alpha
                    )

                # Update policy using the collected episode data
                losses = self.update_policy() # This clears the memory after update
                policy_loss = losses['policy_loss'] if losses else 0
                value_loss = losses['value_loss'] if losses else 0
                entropy = losses['entropy'] if losses else 0
            else:
                # No steps taken in the episode, no policy update
                policy_loss = 0
                value_loss = 0
                entropy = 0
                logger.warning(f"Episode {episodes_completed+1}: No steps taken, no policy update.")


            # Compute entropy of action distribution for logging
            # Handle case where no actions were taken
            if actions_taken:
                action_counts = np.bincount(actions_taken, minlength=self.action_dim)
                most_common_action = np.argmax(action_counts)
                action_frequency = action_counts[most_common_action] / len(actions_taken)
            else:
                most_common_action = -1 # Or some indicator of no action
                action_frequency = 0.0


            # Update episode metrics
            episodes_completed += 1
            episode_rewards.append(episode_reward)
            self.episode_metrics['rewards'].append(episode_reward)
            self.episode_metrics['lengths'].append(episode_steps)
            self.episode_metrics['successes'].append(1 if is_success else 0)
            self.episode_metrics['entropies'].append(entropy) # Use the entropy from the update

            if is_success:
                self.successful_episodes += 1

            # Log episode results
            success_str = "Yes" if is_success else "No"
            timeout_str = " (Timeout)" if reached_max_steps else ""
            logger.info(f"Episode {episodes_completed}/{episodes_to_train} | "
                        f"Reward: {episode_reward:.2f} | "
                        f"Steps: {episode_steps} | "
                        f"Success: {success_str}{timeout_str} | "
                        f"Entropy: {entropy:.4f}")

            # Write to CSV log
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(self.episode_log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp,
                    scenario_name,
                    episodes_completed,
                    round(episode_reward, 2),
                    episode_steps,
                    success_str,
                    round(entropy, 4),
                    round(self.entropy_coef, 4),
                    most_common_action,
                    round(action_frequency, 2)
                ])

            # Save best model based on episode reward
            if episode_reward > best_reward:
                best_reward = episode_reward
                save_path = f"{exp_dir}/best_model.pt"
                self.save(save_path)
                logger.info(f"New best model saved with reward: {best_reward:.2f}")

            # Periodically update reward clipping
            if len(self.reward_history) >= 50:
                self._update_reward_clipping()

            # Create and save plots every 100 episodes or at the end
            if episodes_completed % 100 == 0 or episodes_completed == episodes_to_train:
                # Call the imported plotting function
                plot_rolling_averages(
                    scenario_name=scenario_name,
                    episode_metrics=self.episode_metrics,
                    metrics_dir=self.metrics_dir,
                    logger=logger # Pass the logger instance
                )

            # Log progress to console periodically
            if episodes_completed % 10 == 0:
                # Calculate recent averages carefully to avoid index errors
                recent_episodes_count = min(10, len(self.episode_metrics['rewards']))
                if recent_episodes_count > 0:
                    avg_reward = np.mean(self.episode_metrics['rewards'][-recent_episodes_count:])
                    success_rate = np.mean(self.episode_metrics['successes'][-recent_episodes_count:])
                    logger.info(f"Progress: {episodes_completed}/{episodes_to_train} episodes | "
                                f"Recent Avg Reward: {avg_reward:.2f} | "
                                f"Recent Success Rate: {success_rate:.2%}")
                else:
                     logger.info(f"Progress: {episodes_completed}/{episodes_to_train} episodes")


        # End of training for this scenario
        # After training on a scenario, clear the memory to start fresh for the next one
        self.memory.clear()


        # Compute final statistics
        training_time = time.time() - scenario_start_time
        avg_reward = np.mean(episode_rewards) if episode_rewards else 0.0
        success_rate = self.successful_episodes / episodes_completed if episodes_completed > 0 else 0.0

        # Save final model
        final_path = f"{exp_dir}/final_model.pt"
        self.save(final_path)

        # Create final plots
        # Call the imported plotting function
        plot_rolling_averages(
            scenario_name=scenario_name,
            episode_metrics=self.episode_metrics,
            metrics_dir=self.metrics_dir,
            logger=logger # Pass the logger instance
        )

        # Log completion
        logger.info(f"\nCompleted training on {scenario_name}")
        logger.info(f"Episodes: {episodes_completed}")
        logger.info(f"Training time: {training_time:.1f} seconds")
        logger.info(f"Average reward: {avg_reward:.2f}")
        logger.info(f"Success rate: {success_rate:.2%}")
        logger.info(f"Final model saved to {final_path}")

        # Record scenario performance
        scenario_performance = {
            'scenario': scenario_name,
            'completed': True,
            'episodes': episodes_completed,
            'training_time': training_time,
            'avg_reward': float(avg_reward),
            'best_reward': float(best_reward),
            'successful_episodes': self.successful_episodes,
            'success_rate': float(success_rate)
        }

        # Store performance
        self.scenario_performances[scenario_name] = scenario_performance
        self.scenario_history.append(scenario_name)

        return scenario_performance

    def train_curriculum(self, scenarios, episodes_per_scenario):
        """Train the agent through a curriculum of scenarios."""
        # Initialize experiment directory
        curriculum_dir = f"curriculum_experiments/Checkpoints"
        os.makedirs(curriculum_dir, exist_ok=True)

        # Save curriculum configuration
        config = {
            'scenarios': scenarios,
            'episodes_per_scenario': episodes_per_scenario,
            'timestamp': self.timestamp,
            'hyperparameters': {
                'hidden_size': self.lstm.hidden_dim,
                'lstm_layers': self.lstm.lstm_layers,
                'gamma': self.gamma,
                'gae_lambda': self.gae_lambda, # Keeping GAE
                'clip_ratio': self.clip_ratio, # Keeping clip_ratio
                'value_coef': self.value_coef,
                'entropy_coef': self.entropy_coef,
                'max_grad_norm': self.max_grad_norm,
                'ppo_epochs': self.ppo_epochs, # Keeping original name
                'ppo_batch_size': self.ppo_batch_size, # Keeping original name
                'progress_advantage_alpha': self.progress_advantage_alpha # Added alpha
            }
        }

        with open(f"{curriculum_dir}/config.json", 'w') as f:
            json.dump(config, f, indent=4, cls=NumpyEncoder)

        logger.info(f"\nStarting Curriculum Learning...")
        logger.info(f"Training through {len(scenarios)} scenarios")
        logger.info(f"Scenarios: {scenarios}")

        curriculum_start_time = time.time()
        all_scenario_results = []

        # Main curriculum loop - train on each scenario sequentially
        for i, scenario in enumerate(scenarios):
            logger.info(f"\n\n{'#' * 80}")
            logger.info(f"CURRICULUM STEP {i + 1}/{len(scenarios)}: {scenario}")
            logger.info(f"{'#' * 80}\n")

            # Get episodes for this scenario
            if isinstance(episodes_per_scenario, dict):
                scenario_episodes = episodes_per_scenario.get(scenario, 1000) # Default to 1000 if not specified
            else:
                # If episodes_per_scenario is a single integer, use it for all
                scenario_episodes = episodes_per_scenario


            # Train on this scenario
            scenario_result = self.train_on_scenario(
                scenario_name=scenario,
                episodes_to_train=scenario_episodes
            )

            all_scenario_results.append(scenario_result)

            # Save intermediate curriculum results
            with open(f"{curriculum_dir}/scenario_results.json", 'w') as f:
                json.dump(all_scenario_results, f, indent=4, cls=NumpyEncoder)

            # Save agent after each scenario
            curriculum_checkpoint = f"{curriculum_dir}/checkpoint_after_{scenario}.pt"
            self.save(curriculum_checkpoint)

            logger.info(f"Completed {i + 1}/{len(scenarios)} scenarios in curriculum")

        # Save final model after the entire curriculum
        final_path = f"{curriculum_dir}/final_model.pt"
        self.save(final_path)

        # Compute curriculum statistics
        curriculum_time = time.time() - curriculum_start_time
        completed_scenarios = sum(1 for result in all_scenario_results if result.get('completed', False))

        curriculum_results = {
            'total_scenarios': len(scenarios),
            'completed_scenarios': completed_scenarios,
            'curriculum_time': curriculum_time,
            'scenario_results': all_scenario_results,
            'final_model_path': final_path
        }

        # Save final results
        with open(f"{curriculum_dir}/curriculum_results.json", 'w') as f:
            json.dump(curriculum_results, f, indent=4, cls=NumpyEncoder)

        logger.info(f"\n\n{'=' * 80}")
        logger.info(f"CURRICULUM LEARNING COMPLETED")
        logger.info(f"{'=' * 80}")
        logger.info(f"Completed {completed_scenarios}/{len(scenarios)} scenarios")
        logger.info(f"Total training time: {curriculum_time / 3600:.2f} hours")
        logger.info(f"Final model saved to {final_path}")

        return curriculum_results

    def test(self, scenario_name, num_episodes=50, render=False, deterministic=True):
        """Test the trained agent on a specific scenario."""
        # Setup the environment for testing
        try:
            scenario_name_formatted = "".join(word.capitalize() for word in scenario_name.split("-"))
            # Use render_mode="human" if render is True for gymnasium
            render_mode = "human" if render else None
            test_env = gym.make(f"nasim:{scenario_name_formatted}PO-v0", render_mode=render_mode)


            # Update agent's environment reference
            self.env = test_env

            # Try to get target hosts for this environment
            try:
                if hasattr(test_env, 'scenario') and hasattr(test_env.scenario, 'target_hosts'):
                    self.target_hosts = set(test_env.scenario.target_hosts)
                else:
                    self.target_hosts = None
            except:
                self.target_hosts = None

            # Update agent's observation and action dimensions to match the new environment
            new_obs_dim = self._get_obs_dim(test_env)
            new_action_dim = test_env.action_space.n

            # Handle potential changes in observation or action space dimensions by recreating networks
            # This is similar to the logic in train_on_scenario for handling dimension changes.
            if new_obs_dim != self.obs_dim or new_action_dim != self.action_dim:
                 logger.info(f"Test environment dimensions changed. Adapting agent...")

                 # Store old network states for partial loading
                 old_lstm_state = self.lstm.state_dict()
                 old_a2c_state = self.a2c.state_dict()

                 # Update dimensions
                 self.obs_dim = new_obs_dim
                 self.action_dim = new_action_dim

                 # Recreate networks with new dimensions but same hidden size/layers
                 self.lstm = LSTM(self.obs_dim, self.lstm.hidden_dim, self.lstm.lstm_layers).to(device)
                 self.a2c = A2CNetwork(self.lstm.hidden_dim, self.lstm.hidden_dim, self.action_dim).to(device)

                 # Attempt to partially load old states into new networks
                 try:
                     self.lstm.load_state_dict(old_lstm_state, strict=False)
                     logger.info("Partially loaded old LSTM state into new test network.")
                 except Exception as e:
                     logger.warning(f"Could not partially load old LSTM state during testing adaptation: {e}")

                 try:
                     self.a2c.load_state_dict(old_a2c_state, strict=False)
                     logger.info("Partially loaded old A2C state into new test network.")
                 except Exception as e:
                     logger.warning(f"Could not partially load old A2C state during testing adaptation: {e}")

                 logger.info(f"Agent adapted to {new_obs_dim} obs dims and {new_action_dim} action dims for testing.")

            # Apply scenario-specific settings for testing if needed (optional, depends on if you want test-specific params)
            # self._apply_scenario_settings(scenario_name) # Uncomment if you have test-specific settings


            # Set environment-specific step limit
            # Keeping original step limits
            if 'tiny' in scenario_name or 'tiny-hard' in scenario_name or 'tiny-small' in scenario_name or 'small' in scenario_name or 'small-honeypot' in scenario_name or 'small-linear' in scenario_name:
                max_steps = 1000
            elif 'medium' in scenario_name or 'medium-single-site' in scenario_name or 'medium-multi-site' in scenario_name:
                max_steps = 2000
            else:
                max_steps = 5000  # Default

        except Exception as e:
            logger.error(f"Failed to create test environment: {e}")
            return [] # Return empty list on failure

        logger.info(f"\nTesting agent on {scenario_name} for {num_episodes} episodes")
        logger.info(f"Using deterministic actions: {deterministic}")

        test_rewards = []
        test_lengths = []
        successful_episodes = 0
        timeout_episodes = 0

        for episode in range(num_episodes):
            # Reset environment and agent state
            observation, info = self.env.reset() # Gymnasium returns obs, info
            self.obs_buffer.clear()
            self.hidden = None # Reset LSTM hidden state

            # Get the action mask for the initial state
            current_action_mask = info.get('action_mask', None)

            # Reset episode-specific tracking
            compromised_hosts = set()

            # Track episode metrics
            episode_reward = 0
            episode_length = 0
            actions_taken = []
            reached_max_steps = False

            # Final info from environment
            final_info = {}

            # Run episode
            done = False
            while not done and episode_length < max_steps:
                if render:
                    self.env.render() # Use self.env


                # Select action deterministically during testing, passing the current action mask
                action, _, _ = self.select_action(observation, action_mask=current_action_mask, deterministic=deterministic)
                actions_taken.append(action)

                # Step environment
                try:
                    step_result = self.env.step(action)
                    # Handle both new and old Gym API formats
                    if len(step_result) == 5:  # New Gym API
                        observation, reward, terminated, truncated, info = step_result
                        done = terminated or truncated
                    else:  # Old Gym API
                        observation, reward, done, info = step_result

                    # Update the action mask for the *next* state
                    current_action_mask = info.get('action_mask', None)


                    # Keep track of final info when episode is about to end
                    if done or episode_length == max_steps - 1:
                         # Make a copy of info
                        final_info = info.copy() if info is not None else {}

                    # Update compromised hosts tracking from info
                    if info and 'compromised_hosts' in info:
                        if isinstance(info['compromised_hosts'], (list, tuple, set)):
                            compromised_hosts.update(info['compromised_hosts'])
                        elif info['compromised_hosts'] is not None: # Handle non-iterable but not None
                             compromised_hosts.add(info['compromised_hosts'])

                except Exception as e:
                    logger.error(f"Test step error: {e}")
                    # Terminate episode on error
                    done = True
                    observation = observation # Stay in current state or handle appropriately
                    reward = 0 # Assign a penalty or 0 reward on error
                    info = {} # Clear info or add error info
                    current_action_mask = None # Reset mask on error


                # Update stats
                episode_reward += reward
                episode_length += 1

            # Check if episode timed out
            if episode_length >= max_steps:
                reached_max_steps = True
                timeout_episodes += 1

            # Determine if episode was successful (target host compromised)
            # Use the final_info captured just before the episode ended
            is_success = self.check_success(final_info)

            # If we have target hosts defined, check if any were compromised
            # This check uses the accumulated compromised_hosts during the episode
            if not is_success and self.target_hosts and compromised_hosts:
                is_success = bool(compromised_hosts.intersection(self.target_hosts))

            # Store results
            test_rewards.append(episode_reward)
            test_lengths.append(episode_length)

            if is_success:
                successful_episodes += 1

            # Analyze action distribution for this episode
            # Handle case where no actions were taken
            if actions_taken:
                action_counts = np.bincount(actions_taken, minlength=self.action_dim)
                most_common = np.argmax(action_counts)
            else:
                most_common = -1 # Or some indicator of no action


            # Add timeout information to output
            timeout_str = " (Timeout)" if reached_max_steps else ""
            logger.info(f"Test Episode {episode + 1}/{num_episodes} | " +
                        f"Reward: {episode_reward:.2f} | " +
                        f"Length: {episode_length} | " +
                        f"Success: {'Yes' if is_success else 'No'}{timeout_str}")

        # Print average results
        avg_reward = np.mean(test_rewards) if test_rewards else 0.0
        avg_length = np.mean(test_lengths) if test_lengths else 0.0
        success_rate = successful_episodes / num_episodes if num_episodes > 0 else 0.0
        timeout_rate = timeout_episodes / num_episodes if num_episodes > 0 else 0.0


        logger.info(f"\nTest Results Summary:")
        logger.info(f"Average Reward: {avg_reward:.2f}")
        logger.info(f"Average Episode Length: {avg_length:.1f}")
        logger.info(f"Success Rate: {success_rate:.2%} ({successful_episodes}/{num_episodes})")
        logger.info(f"Timeout Rate: {timeout_rate:.2%} ({timeout_episodes}/{num_episodes})")

        # Close the environment
        self.env.close() # Use self.env

        return test_rewards

    def save(self, path):
        """Save the agent's networks and training state."""
        torch.save({
            'lstm_state_dict': self.lstm.state_dict(),
            'a2c_state_dict': self.a2c.state_dict(), # Using A2CNetwork
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'obs_dim': self.obs_dim,
            'action_dim': self.action_dim,
            'hidden_dim': self.lstm.hidden_dim,
            'lstm_layers': self.lstm.lstm_layers,
            'clip_ratio': self.clip_ratio, # Keeping clip_ratio
            'entropy_coef': self.entropy_coef,
            'reward_clip': self.reward_clip,
            'episode_count': self.episode_count,
            'scenario_history': self.scenario_history,
            'scenario_performances': self.scenario_performances,
            'ppo_epochs': self.ppo_epochs, # Keeping original name
            'ppo_batch_size': self.ppo_batch_size, # Keeping original name
            'progress_advantage_alpha': self.progress_advantage_alpha # Added alpha
        }, path)

    def load(self, path):
        """Load the agent's networks and training state, handling dimension changes."""
        if not os.path.exists(path):
            logger.error(f"Checkpoint file not found at: {path}")
            raise FileNotFoundError(f"Checkpoint file not found at: {path}")

        logger.info(f"Loading checkpoint from: {path}")
        # Add the FutureWarning suppression only if necessary, or address it
        # The recommendation is to use weights_only=True if possible.
        # If you MUST load arbitrary code via pickle, be absolutely sure of the source.
        # For now, keeping original behavior but user should be aware.
        checkpoint = torch.load(path, map_location=device)  # Add weights_only=True if appropriate

        # --- Load model architecture parameters ---
        # Use .get() with defaults to handle older checkpoints missing keys
        ckpt_obs_dim = checkpoint.get('obs_dim')
        ckpt_action_dim = checkpoint.get('action_dim')
        ckpt_hidden_dim = checkpoint.get('hidden_dim')
        ckpt_lstm_layers = checkpoint.get('lstm_layers')

        # Validate essential parameters were found in checkpoint
        if None in [ckpt_obs_dim, ckpt_action_dim, ckpt_hidden_dim, ckpt_lstm_layers]:
            logger.warning("Checkpoint missing one or more dimension/layer parameters. Using current agent defaults.")
            # Use current agent attributes as fallback if checkpoint is missing info
            ckpt_obs_dim = ckpt_obs_dim if ckpt_obs_dim is not None else self.obs_dim
            ckpt_action_dim = ckpt_action_dim if ckpt_action_dim is not None else self.action_dim
            ckpt_hidden_dim = ckpt_hidden_dim if ckpt_hidden_dim is not None else self.lstm.hidden_dim
            ckpt_lstm_layers = ckpt_lstm_layers if ckpt_lstm_layers is not None else self.lstm.lstm_layers

        # --- Check if network architecture needs recreation ---
        # Compare checkpoint dimensions with the *current* agent's network dimensions
        recreate_networks = False
        if (ckpt_obs_dim != self.obs_dim or
                ckpt_action_dim != self.action_dim or
                ckpt_hidden_dim != self.lstm.hidden_dim or
                ckpt_lstm_layers != self.lstm.lstm_layers):
            recreate_networks = True
            logger.info(
                f"Recreating networks with dims from checkpoint: obs={ckpt_obs_dim}, action={ckpt_action_dim}, hidden={ckpt_hidden_dim}, lstm_layers={ckpt_lstm_layers}")
            # Update agent's dimensions based on the checkpoint being loaded
            self.obs_dim = ckpt_obs_dim
            self.action_dim = ckpt_action_dim

        # --- Load Network States ---
        if recreate_networks:
            # Create new networks with dimensions from the checkpoint
            # Note: hidden_dim and lstm_layers should come from ckpt_ variables now
            self.lstm = LSTM(self.obs_dim, ckpt_hidden_dim, ckpt_lstm_layers).to(device)
            self.a2c = A2CNetwork(ckpt_hidden_dim, ckpt_hidden_dim, self.action_dim).to(device)

            # --- CORRECTED LOGIC ---
            # Load state dicts from the CHECKPOINT data, not from old_state
            # Use strict=False to allow loading even if some keys are unexpectedly missing/extra
            # (e.g., if network definition changed slightly between saving and loading)
            if 'lstm_state_dict' in checkpoint:
                try:
                    missing_keys, unexpected_keys = self.lstm.load_state_dict(checkpoint['lstm_state_dict'],
                                                                              strict=False)
                    if missing_keys:
                        logger.warning(f"LSTM loaded with missing keys: {missing_keys}")
                    if unexpected_keys:
                        logger.warning(f"LSTM loaded with unexpected keys: {unexpected_keys}")
                    logger.info("Loaded LSTM state from checkpoint (strict=False).")
                except Exception as e:
                    logger.error(f"Error loading LSTM state dict from checkpoint: {e}", exc_info=True)
            else:
                logger.warning("Checkpoint missing 'lstm_state_dict'. LSTM network initialized randomly.")

            if 'a2c_state_dict' in checkpoint:
                try:
                    missing_keys, unexpected_keys = self.a2c.load_state_dict(checkpoint['a2c_state_dict'], strict=False)
                    if missing_keys:
                        logger.warning(f"A2C loaded with missing keys: {missing_keys}")
                    if unexpected_keys:
                        logger.warning(f"A2C loaded with unexpected keys: {unexpected_keys}")
                    logger.info("Loaded A2C state from checkpoint (strict=False).")
                except Exception as e:
                    logger.error(f"Error loading A2C state dict from checkpoint: {e}", exc_info=True)
            else:
                logger.warning("Checkpoint missing 'a2c_state_dict'. A2C network initialized randomly.")
            # --- END CORRECTION ---

            # --- Optimizer Handling ---
            # Create a new optimizer because network parameters changed.
            # Try to get LR from checkpoint, otherwise use a default.
            current_lr = self.optimizer.param_groups[0]['lr']  # Default to existing LR
            if 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict']['param_groups']:
                current_lr = checkpoint['optimizer_state_dict']['param_groups'][0]['lr']

            self.optimizer = optim.Adam(
                list(self.lstm.parameters()) + list(self.a2c.parameters()),
                lr=current_lr
            )
            logger.info(
                f"Created new optimizer with learning rate: {current_lr}. Optimizer state not transferred due to dimension change.")

        else:
            # Dimensions match, load state dicts normally (use strict=True by default)
            try:
                self.lstm.load_state_dict(checkpoint['lstm_state_dict'])
                self.a2c.load_state_dict(checkpoint['a2c_state_dict'])
                logger.info("Loaded LSTM and A2C states (strict=True).")
            except KeyError as e:
                logger.error(f"Checkpoint missing required key: {e}. Cannot load model state.", exc_info=True)
                # Depending on desired behavior, you might raise the error or allow random init
                raise RuntimeError(f"Checkpoint missing required key: {e}") from e
            except Exception as e:
                logger.error(f"Error loading model state dicts: {e}", exc_info=True)
                raise RuntimeError(f"Error loading model state dicts: {e}") from e

            # Load optimizer state ONLY if dimensions match
            if 'optimizer_state_dict' in checkpoint:
                try:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    logger.info("Loaded optimizer state.")
                except Exception as e:
                    logger.warning(f"Could not load optimizer state dict, creating new one. Error: {e}", exc_info=True)
                    # Recreate optimizer if loading fails, using loaded LR if possible
                    current_lr = checkpoint['optimizer_state_dict']['param_groups'][0]['lr'] if \
                    checkpoint['optimizer_state_dict']['param_groups'] else self.optimizer.param_groups[0]['lr']
                    self.optimizer = optim.Adam(
                        list(self.lstm.parameters()) + list(self.a2c.parameters()),
                        lr=current_lr
                    )

        # --- Load scheduler if available ---
        if 'scheduler_state_dict' in checkpoint:
            try:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                logger.info("Loaded scheduler state.")
            except Exception as e:
                logger.warning(f"Could not load scheduler state dict: {e}")

        # --- Load hyperparameters (use .get() with defaults) ---
        self.clip_ratio = checkpoint.get('clip_ratio', self.clip_ratio)
        self.entropy_coef = checkpoint.get('entropy_coef', self.entropy_coef)
        self.reward_clip = checkpoint.get('reward_clip', self.reward_clip)
        self.ppo_epochs = checkpoint.get('ppo_epochs', self.ppo_epochs)
        self.ppo_batch_size = checkpoint.get('ppo_batch_size', self.ppo_batch_size)
        self.progress_advantage_alpha = checkpoint.get('progress_advantage_alpha', self.progress_advantage_alpha)

        # --- Load training state (use .get() with defaults) ---
        self.episode_count = checkpoint.get('episode_count', 0)
        self.scenario_history = checkpoint.get('scenario_history', [])
        self.scenario_performances = checkpoint.get('scenario_performances', {})

        logger.info(f"Agent loaded successfully from {path}")
