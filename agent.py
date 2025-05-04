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


# Assume these utility modules are in the same directory or accessible
# Make sure these files exist and contain the necessary definitions
try:
    # Import the ACTUAL modules
    from utils import device, logger, NumpyEncoder, PPOTransition
    from networks import LSTM, A2CNetwork
    from memory import A2CMemory
    from plot import plot_rolling_averages
except ImportError as e:
    print(f"Error importing utility modules: {e}")
    print("Please ensure utils.py, networks.py, memory.py, and plot.py are available.")
    # Define dummy logger if not imported (as a basic fallback for script loading)
    import logging
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)
    # Define dummy device if torch not fully functional or utils missing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




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
        # Ensure ACTUAL LSTM and A2CNetwork are used (dummy assignments removed)
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
        # Ensure ACTUAL A2CMemory is used (dummy assignment removed)
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

        # Timestamp for the entire agent run (used for folder naming)
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # --- Logging setup REMOVED from __init__ ---

        # Log model size and initial GPU memory usage
        self._log_model_info()

    def _log_model_info(self):
        """Log model size and initial GPU memory utilization."""
        total_params = sum(p.numel() for p in self.lstm.parameters() if p.requires_grad)
        total_params += sum(p.numel() for p in self.a2c.parameters() if p.requires_grad) # Using A2CNetwork
        logger.info(f"Model initialized with {total_params:,} trainable parameters")
        if torch.cuda.is_available():
            # Corrected typo in reserved memory calculation if necessary
            logger.info(f"Initial GPU memory: {torch.cuda.memory_allocated() / 1024 ** 2:.1f}MB / "
                        f"{torch.cuda.memory_reserved() / 1024 ** 2:.1f}MB")


    def _get_obs_dim(self, env):
        """Compute observation dimension from environment."""
        # Keep original logic
        if hasattr(env.observation_space, 'spaces'):
            # Ensure consistent order if original code relied on it (e.g., sorting keys)
            # If original just iterated, use that:
            # return sum(np.prod(space.shape) for space in env.observation_space.spaces.values())
            # If original sorted:
            return sum(np.prod(space.shape) for k, space in sorted(env.observation_space.spaces.items()))
        elif isinstance(env.observation_space, gym.spaces.Tuple):
             return sum(np.prod(space.shape) for space in env.observation_space.spaces)
        # Default for Box, etc.
        return np.prod(env.observation_space.shape)


    def _apply_scenario_settings(self, scenario_name):
        """Apply environment-specific hyperparameters."""
        # Keep original logic
        if scenario_name in self.scenario_settings:
            settings = self.scenario_settings[scenario_name]
            logger.info(f"Applying settings for scenario: {scenario_name}") # Added log for clarity

            if 'entropy_coef' in settings:
                self.entropy_coef = settings['entropy_coef']
                logger.info(f"  - Setting entropy coefficient to {self.entropy_coef}")

            if 'learning_rate' in settings:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = settings['learning_rate']
                logger.info(f"  - Setting learning rate to {settings['learning_rate']}")

            if 'clip_ratio' in settings:
                self.clip_ratio = settings['clip_ratio']
                logger.info(f"  - Setting clip ratio to {self.clip_ratio}")

            if 'progress_advantage_alpha' in settings:
                self.progress_advantage_alpha = settings['progress_advantage_alpha']
                logger.info(f"  - Setting progress advantage alpha to {self.progress_advantage_alpha}")

            # Reset reward clipping for new scenario
            self.reward_clip = [-20, 20]
            logger.info(f"  - Resetting reward clip range to {self.reward_clip}") # Added log
        else:
            # Default settings if scenario not explicitly configured
            logger.warning(f"No specific settings found for scenario '{scenario_name}'. Using defaults.") # Added log
            self.entropy_coef = 0.01
            self.clip_ratio = 0.2
            self.progress_advantage_alpha = 1.0 # Default alpha
            # Ensure LR is set to default if not specified (might already be handled by init)
            # Check if optimizer exists and has param groups before accessing LR
            if hasattr(self, 'optimizer') and self.optimizer.param_groups:
                 # Check if LR needs resetting (e.g., if previous scenario changed it)
                 # This part depends on whether you want settings to persist or reset fully
                 # Resetting to a default might be safer if not specified:
                 # self.optimizer.param_groups[0]['lr'] = 0.0003 # Example default LR
                 pass # Keep current LR if not specified in defaults


    def _update_entropy_coefficient(self):
        """Dynamically adjust entropy coefficient based on performance."""
        # Keep original logic
        if len(self.episode_metrics['successes']) < 10: # Original threshold
            return

        recent_successes = self.episode_metrics['successes'][-20:] # Original window
        # Handle division by zero if len is 0 (though check above should prevent)
        recent_success_rate = sum(recent_successes) / len(recent_successes) if recent_successes else 0.0
        recent_entropies = self.episode_metrics['entropies'][-20:] # Original window
        avg_entropy = sum(recent_entropies) / max(1, len(recent_entropies))

        target_entropy_coef = self.entropy_coef # Initialize with current value

        # Original logic for adjusting target
        if recent_success_rate < 0.1:
            target_entropy_coef = min(self.max_entropy_coef, self.entropy_coef * 2.0)
        elif recent_success_rate < 0.3:
            target_entropy_coef = min(self.max_entropy_coef, self.entropy_coef * 1.2)
        elif recent_success_rate > 0.8 and avg_entropy < 0.5: # Original entropy threshold
            target_entropy_coef = max(self.min_entropy_coef, self.entropy_coef * 0.9)
        # else: target_entropy_coef remains self.entropy_coef


        # Original smooth update logic
        self.entropy_coef = 0.95 * self.entropy_coef + 0.05 * target_entropy_coef
        self.entropy_coef = max(self.min_entropy_coef, min(self.max_entropy_coef, self.entropy_coef))

        # Original logging condition (log if changed)
        # Use a small tolerance for floating point comparison
        if abs(self.entropy_coef - target_entropy_coef) > 1e-9: # Check if it actually changed
            logger.info(
                f"Adjusted entropy coefficient to {self.entropy_coef:.6f} (success rate: {recent_success_rate:.2f}, avg entropy: {avg_entropy:.4f})") # Log avg entropy too


    def _update_reward_clipping(self):
        """Dynamically adjust reward clipping based on recent rewards."""
        # Keep original logic
        if len(self.reward_history) < 50: # Original threshold
            return

        min_reward = np.percentile(list(self.reward_history), 10) # Original percentile
        max_reward = np.percentile(list(self.reward_history), 90) # Original percentile

        min_reward = max(-200, min_reward) # Original bounds
        max_reward = min(1000, max_reward) # Original bounds

        # Original smooth update logic
        new_min_clip = 0.9 * self.reward_clip[0] + 0.1 * min_reward
        new_max_clip = 0.9 * self.reward_clip[1] + 0.1 * max_reward

        # Log only if changed significantly (original behavior implied)
        if abs(new_min_clip - self.reward_clip[0]) > 0.1 or abs(new_max_clip - self.reward_clip[1]) > 0.1:
             self.reward_clip[0] = new_min_clip
             self.reward_clip[1] = new_max_clip



    def preprocess_observation(self, observation):
        """Convert observation to tensor and add to sequence buffer."""
        # Keep original logic exactly
        flat_obs = None
        try:
            if isinstance(observation, dict):
                flat_parts = []
                # Use original key iteration order if it mattered (e.g., simple .items())
                # If sorted order was used:
                for k in sorted(observation.keys()): # Assuming sorted was original
                    v = observation[k]
                # for k, v in observation.items(): # If simple iteration was original
                    if isinstance(v, np.ndarray):
                        flat_parts.append(v.flatten())
                    elif isinstance(v, (int, float, np.number)):
                        flat_parts.append(np.array([float(v)]))
                    elif isinstance(v, (list, tuple)):
                         flat_parts.append(np.array(v, dtype=float).flatten())
                    else:
                        # Keep original warning/handling
                        logger.warning(f"Unsupported observation dict value type for key '{k}': {type(v)}. Attempting conversion.")
                        try:
                            flat_parts.append(np.array([float(v)]).flatten())
                        except:
                            logger.warning(f"Failed to convert unsupported type for key '{k}'. Using zero vector.")
                            # Use original fallback shape if specified, else default to 1
                            fallback_shape = getattr(self.env.observation_space[k], 'shape', (1,)) if hasattr(self.env.observation_space, 'spaces') and k in self.env.observation_space.spaces else (1,)
                            flat_parts.append(np.zeros(np.prod(fallback_shape)))


                if flat_parts:
                    # Ensure original dtype casting and filtering
                    flat_parts = [part.astype(np.float32) for part in flat_parts if part.size > 0]
                    if flat_parts:
                        flat_obs = np.concatenate(flat_parts)
                    else:
                        flat_obs = np.zeros(self.obs_dim, dtype=np.float32) # Original fallback

            elif isinstance(observation, (tuple, list)):
                flat_parts = []
                for i, item in enumerate(observation): # Use enumerate if original did
                    if isinstance(item, (int, float, np.number)):
                        flat_parts.append(np.array([float(item)]))
                    elif isinstance(item, np.ndarray):
                        flat_parts.append(item.flatten())
                    elif isinstance(item, (list, tuple)):
                        flat_parts.append(np.array(item, dtype=float).flatten())
                    else:
                        # Keep original warning/handling
                        logger.warning(f"Unsupported observation list/tuple element type: {type(item)}. Attempting conversion.")
                        try:
                            flat_parts.append(np.array([float(item)]).flatten())
                        except:
                            logger.warning(f"Failed to convert unsupported type. Using zero vector.")
                            # Use original fallback shape if specified, else default to 1
                            fallback_shape = self.env.observation_space[i].shape if isinstance(self.env.observation_space, gym.spaces.Tuple) and i < len(self.env.observation_space.spaces) else (1,)
                            flat_parts.append(np.zeros(np.prod(fallback_shape)))

                if flat_parts:
                    # Ensure original dtype casting and filtering
                    flat_parts = [part.astype(np.float32) for part in flat_parts if part.size > 0]
                    if flat_parts:
                        flat_obs = np.concatenate(flat_parts)
                    else:
                        flat_obs = np.zeros(self.obs_dim, dtype=np.float32) # Original fallback

            elif isinstance(observation, np.ndarray):
                flat_obs = observation.flatten().astype(np.float32) # Original casting
            else:
                # Keep original primitive type handling
                try:
                    flat_obs = np.array([float(observation)], dtype=np.float32).flatten()
                except:
                    logger.warning(f"Unsupported observation type: {type(observation)}. Using zero vector.")
                    flat_obs = np.zeros(self.obs_dim, dtype=np.float32) # Original fallback

        except Exception as e:
            # Keep original error handling
            logger.error(f"Critical error during observation preprocessing: {e}. Using zero vector.")
            flat_obs = np.zeros(self.obs_dim, dtype=np.float32)


        # Keep original check for None or empty array
        if flat_obs is None or flat_obs.size == 0:
             logger.warning("Observation preprocessing resulted in empty array. Using zero vector.")
             flat_obs = np.zeros(self.obs_dim, dtype=np.float32)


        # Keep original dimension mismatch handling (padding/truncating)
        if len(flat_obs) != self.obs_dim:
            # logger.warning(f"Observation dimension mismatch: Expected {self.obs_dim}, got {len(flat_obs)}. Padding or truncating.") # Keep commented if original was
            if len(flat_obs) < self.obs_dim:
                padded_obs = np.pad(flat_obs, (0, self.obs_dim - len(flat_obs)), 'constant')
                flat_obs = padded_obs
            else:
                flat_obs = flat_obs[:self.obs_dim]


        # Keep original normalization logic exactly
        max_val = np.max(flat_obs) if flat_obs.size > 0 else 100.0 # Original default
        # Ensure calculation is identical
        norm_obs = np.clip(flat_obs, 0, max_val) / (max_val + 1e-8) * 100.0
        norm_obs = np.clip(norm_obs, 0, 100) / 100.0 # Original two-step process

        obs_tensor = torch.tensor(norm_obs, dtype=torch.float32, device=device) # Use norm_obs here

        # Keep original buffer logic
        self.obs_buffer.append(obs_tensor)

        if len(self.obs_buffer) < self.sequence_length:
            padding = [torch.zeros_like(obs_tensor) for _ in range(self.sequence_length - len(self.obs_buffer))]
            padded_buffer = padding + list(self.obs_buffer) # Original padding order
        else:
            padded_buffer = list(self.obs_buffer)

        sequence = torch.stack(padded_buffer).unsqueeze(0) # Original final shape

        return sequence


    def select_action(self, observation, action_mask=None, deterministic=False):
        """Select action based on current policy, with optional action masking."""
        # Keep original logic exactly
        with torch.no_grad():
            state_seq = self.preprocess_observation(observation)

            self.lstm.lstm.flatten_parameters()

            lstm_features, next_hidden = self.lstm(state_seq, self.hidden) # Use next_hidden

            # --- Critical: Update the agent's hidden state ---
            self.hidden = next_hidden # Assign the returned hidden state
            # -------------------------------------------------


            value, policy_logits, _ = self.a2c(lstm_features)

            if action_mask is not None:
                # Original masking logic
                mask_tensor = torch.tensor(action_mask, dtype=torch.bool, device=device).unsqueeze(0)
                # Check shape mismatch IF original code did
                # if mask_tensor.shape != policy_logits.shape: ...
                policy_logits[~mask_tensor] = float('-inf')

            # Check for all masked actions IF original code did
            # if torch.all(policy_logits == float('-inf')): ... handle fallback ...
            # else:
            dist = Categorical(logits=policy_logits)

            if deterministic:
                action = dist.probs.argmax(dim=-1)
            else:
                action = dist.sample()

            log_prob = dist.log_prob(action)

            return action.item(), log_prob.item(), value.item()


    def clip_reward(self, reward):
        """Clip reward to handle extreme values."""
        # Keep original logic
        return max(self.reward_clip[0], min(self.reward_clip[1], reward))


    def update_policy(self):
        """Update policy using PPO-like update with specialized advantage."""
        # Keep original logic exactly
        if len(self.memory.states) == 0: # Check original memory attribute used (e.g., states or rewards)
            # Use original warning or return value if different
            # logger.warning("Policy update skipped: No data in memory.") # Keep original warning if present
            return {'policy_loss': 0, 'value_loss': 0, 'entropy': 0} # Keep original return

        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        update_count = 0

        # Original PPO update loop structure
        for _ in range(self.ppo_epochs): # Use original number of epochs
            # Get batches using original memory method
            for states_batch, actions_batch, old_log_probs_batch, returns_batch, advantages_batch in self.memory.get_batches():

                self.optimizer.zero_grad()

                # Original LSTM processing
                self.lstm.lstm.flatten_parameters()
                lstm_features_batch, _ = self.lstm(states_batch, None)

                # Original A2C evaluation
                new_log_probs, values, entropy = self.a2c.evaluate_actions(lstm_features_batch, actions_batch)

                # Original PPO Ratio
                ratio = torch.exp(new_log_probs - old_log_probs_batch)

                # Original Surrogate Loss
                surr1 = ratio * advantages_batch
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages_batch
                policy_loss = -torch.min(surr1, surr2).mean()

                # Original Value Loss (Clipped)
                # Ensure original value shape handling (squeeze or not)
                values_squeezed = values.squeeze(-1) # Assuming original squeezed here
                value_pred_clipped = values_squeezed + torch.clamp( # Check original calculation order
                    values_squeezed - returns_batch, # Original difference calculation
                    -self.clip_ratio, self.clip_ratio
                )
                # Check original MSE calculation (e.g., F.mse_loss or (a-b)**2).mean()
                value_loss_unclipped = F.mse_loss(values_squeezed, returns_batch)
                value_loss_clipped = F.mse_loss(value_pred_clipped, returns_batch) # Use value_pred_clipped here
                value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped) # Original max logic

                # Original Entropy Bonus
                entropy_loss = -entropy.mean()

                # Original Total Loss
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                # Original Backpropagation
                loss.backward()

                # Original Gradient Clipping
                nn.utils.clip_grad_norm_(
                    list(self.lstm.parameters()) + list(self.a2c.parameters()),
                    self.max_grad_norm
                )

                # Original Optimizer Step
                self.optimizer.step()

                # Original Loss Tracking
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                update_count += 1

        # Original call to update entropy coefficient (if present here)
        # self._update_entropy_coefficient() # Keep if original was here

        # Original memory clear location
        self.memory.clear()

        # Original return logic
        if update_count > 0:
            avg_policy_loss = total_policy_loss / update_count
            avg_value_loss = total_value_loss / update_count
            avg_entropy = total_entropy / update_count
            return {
                'policy_loss': avg_policy_loss,
                'value_loss': avg_value_loss,
                'entropy': avg_entropy
            }
        return {'policy_loss': 0, 'value_loss': 0, 'entropy': 0} # Original default return


    def check_success(self, info):
        """Determine if the episode was successful based on environment info."""
        # Keep original logic exactly
        if info is None:
            return False

        # Check for success flag in info (use .get for safety if original didn't)
        if info.get('success', False): # Use .get if original might have missing keys
            return True

        # Check for goal_reached flag (use .get for safety if original didn't)
        if info.get('goal_reached', False):
            return True

        # Check if any target hosts were compromised
        if self.target_hosts and 'compromised_hosts' in info:
            # Use original handling of compromised_hosts type and intersection
            compromised = set(info.get('compromised_hosts', [])) # Keep original .get default
            # Ensure original intersection logic
            return bool(compromised.intersection(self.target_hosts))

        # Check for general compromised flag (use .get for safety if original didn't)
        if info.get('compromised', False) and info['compromised']: # Keep original check
            return True

        # Fallback if no other indicators available
        return False


    def train_on_scenario(self, scenario_name, episodes_to_train):
        """Train the agent on a specific scenario for a fixed number of episodes."""
        # Initialize scenario (Keep original logs)
        logger.info(f"\n{'=' * 50}")
        logger.info(f"Starting training on scenario: {scenario_name}")

        # --- Logging Setup MOVED HERE ---
        self.metrics_dir = f"metrics_Plot/Training/{scenario_name}"
        os.makedirs(self.metrics_dir, exist_ok=True)
        self.episode_log_file = f"{self.metrics_dir}/episode_logs_{scenario_name}.csv" # Use scenario name in file
        try: # Add error handling for file opening (Write mode 'w')
            with open(self.episode_log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                # Use original CSV header columns
                writer.writerow(['Timestamp', 'Scenario', 'Episode', 'Reward', 'Steps', 'Success',
                                 'Entropy', 'EntropyCoef', 'MostCommonAction', 'ActionFrequency'])
            logger.info(f"Logging episode data to: {self.episode_log_file}") # Log file path
        except IOError as e:
             logger.error(f"Failed to create or write header to log file {self.episode_log_file}: {e}")
             # Decide how to handle this error - maybe return failure?
             return { # Example error return
                 'scenario': scenario_name, 'completed': False, 'error': str(e),
                 'episodes': 0, 'avg_reward': 0.0, 'successful_episodes': 0
             }
        # --- End Logging Setup ---


        # Update current scenario
        self.current_scenario = scenario_name

        # Create environment for this scenario (Keep original logic)
        try:
            scenario_name_formatted = "".join(word.capitalize() for word in scenario_name.split("-"))
            new_env = gym.make(f"nasim:{scenario_name_formatted}PO-v0")
            self.env = new_env

            try: # Keep original target host fetching
                if hasattr(new_env, 'scenario') and hasattr(new_env.scenario, 'target_hosts'):
                    self.target_hosts = set(new_env.scenario.target_hosts)
                else:
                    self.target_hosts = None
            except:
                self.target_hosts = None

            # Apply scenario-specific settings (Keep original call)
            self._apply_scenario_settings(scenario_name)

            # Get new observation and action dimensions (Keep original calls)
            new_obs_dim = self._get_obs_dim(new_env)
            new_action_dim = new_env.action_space.n

            # Log environment information (Keep original logs)
            logger.info(f"Scenario: {scenario_name}")
            logger.info(f"Observation Space: {new_env.observation_space}")
            logger.info(f"Observation Dim: {new_obs_dim}")
            logger.info(f"Action Space: {new_env.action_space}")
            logger.info(f"Action Dim: {new_action_dim}")
            if self.target_hosts:
                logger.info(f"Target Hosts: {self.target_hosts}")

            # Handle potential changes in dimensions (Keep original logic)
            if new_obs_dim != self.obs_dim or new_action_dim != self.action_dim:
                logger.info(f"Environment dimensions changed. Transferring knowledge...")
                old_optimizer_state_dict = self.optimizer.state_dict()
                old_lstm = self.lstm
                old_a2c = self.a2c
                old_obs_dim = self.obs_dim
                old_action_dim = self.action_dim
                self.obs_dim = new_obs_dim
                self.action_dim = new_action_dim
                self.lstm = LSTM(self.obs_dim, self.lstm.hidden_dim, self.lstm.lstm_layers).to(device)
                self.a2c = A2CNetwork(self.lstm.hidden_dim, self.lstm.hidden_dim, self.action_dim).to(device)

                with torch.no_grad(): # Keep original weight transfer logic
                    for i in range(min(self.lstm.lstm_layers, old_lstm.lstm_layers)):
                         self.lstm.lstm.__getattr__(f'weight_hh_l{i}').copy_(old_lstm.lstm.__getattr__(f'weight_hh_l{i}'))
                         self.lstm.lstm.__getattr__(f'bias_hh_l{i}').copy_(old_lstm.lstm.__getattr__(f'bias_hh_l{i}'))
                         self.lstm.lstm.__getattr__(f'weight_ih_l{i}').copy_(old_lstm.lstm.__getattr__(f'weight_ih_l{i}'))
                         self.lstm.lstm.__getattr__(f'bias_ih_l{i}').copy_(old_lstm.lstm.__getattr__(f'bias_ih_l{i}'))

                    min_features = min(old_obs_dim, new_obs_dim)
                    if min_features > 0:
                        # Ensure original indexing/slicing for input layer transfer
                        self.lstm.input_layer[0].weight.data[:, :min_features].copy_(
                            old_lstm.input_layer[0].weight.data[:, :min_features]
                        )
                        if hasattr(self.lstm.input_layer[0], 'bias') and hasattr(old_lstm.input_layer[0], 'bias'):
                            self.lstm.input_layer[0].bias.data.copy_(old_lstm.input_layer[0].bias.data)

                    self.a2c.shared.load_state_dict(old_a2c.shared.state_dict())
                    self.a2c.value_head.load_state_dict(old_a2c.value_head.state_dict())

                    min_actions = min(old_action_dim, new_action_dim)
                    if min_actions > 0:
                        # Ensure original indexing/slicing for policy head transfer
                        self.a2c.policy_head[-1].weight.data[:min_actions, :].copy_(
                            old_a2c.policy_head[-1].weight.data[:min_actions, :]
                        )
                        self.a2c.policy_head[-1].bias.data[:min_actions].copy_(
                            old_a2c.policy_head[-1].bias.data[:min_actions]
                        )

                # Keep original optimizer recreation logic
                new_optimizer = optim.Adam(
                    list(self.lstm.parameters()) + list(self.a2c.parameters()),
                    lr=self.optimizer.param_groups[0]['lr']
                )
                # Keep original logic regarding loading old optimizer state (i.e., NOT loading it here)
                self.optimizer = new_optimizer

                logger.info(
                    f"Knowledge transferred from {old_obs_dim}→{new_obs_dim} obs dims and {old_action_dim}→{new_action_dim} action dims")

                # Keep original buffer/hidden state reset
                self.obs_buffer.clear()
                self.hidden = None

        except Exception as e: # Keep original error handling
            logger.error(f"Failed to create environment for scenario {scenario_name}: {e}")
            return {
                'scenario': scenario_name, 'completed': False, 'error': str(e),
                'episodes': 0, 'avg_reward': 0.0, 'successful_episodes': 0
            } # Keep original return format

        # Initialize training variables (Keep original)
        episodes_completed = 0 # Use original variable name
        episode_rewards = []
        best_reward = -float('inf')

        # Create scenario directory for checkpoints (Keep original path structure)
        exp_dir = f"scenario_checkpoints/{scenario_name}" # Original path
        os.makedirs(exp_dir, exist_ok=True)

        # Reset stats for this scenario (Keep original resets)
        self.successful_episodes = 0
        self.episode_metrics['rewards'] = []
        self.episode_metrics['lengths'] = []
        self.episode_metrics['successes'] = []
        self.episode_metrics['entropies'] = []
        self.reward_history.clear()

        scenario_start_time = time.time()

        logger.info(f"Training on {scenario_name} for {episodes_to_train} episodes")

        # Set max steps per episode (Keep original logic)
        if 'tiny' in scenario_name or 'small' in scenario_name or 'huge' in scenario_name:
            max_steps_per_episode = 1000
        elif 'medium' in scenario_name:
            max_steps_per_episode = 2000
        else:
            max_steps_per_episode = 1000

        logger.info(f"Setting max steps per episode to {max_steps_per_episode}")

        # Training loop (Keep original structure)
        while episodes_completed < episodes_to_train:
            # Reset environment and agent state (Keep original)
            obs, info = self.env.reset()
            self.obs_buffer.clear()
            self.hidden = None
            self.memory._initial_info = info
            self.memory.clear()

            current_action_mask = info.get('action_mask', None) # Keep original default

            # Reset episode tracking (Keep original)
            compromised_hosts = set()
            done = False
            episode_reward = 0
            episode_steps = 0
            actions_taken = []
            reached_max_steps = False
            is_success = False
            final_info = {}

            # Inner episode loop (Keep original structure)
            while not done and episode_steps < max_steps_per_episode:
                # Select action (Keep original call)
                action, log_prob, value = self.select_action(obs, action_mask=current_action_mask)
                actions_taken.append(action)

                # Step environment (Keep original logic and error handling)
                try:
                    step_result = self.env.step(action)
                    if len(step_result) == 5:
                        next_obs, reward, terminated, truncated, info = step_result
                        done = terminated or truncated
                    else:
                        next_obs, reward, done, info = step_result

                    current_action_mask = info.get('action_mask', None)

                    if done or episode_steps == max_steps_per_episode - 1:
                        final_info = info.copy() if info is not None else {}

                    if info and 'compromised_hosts' in info: # Keep original compromised host tracking
                        if isinstance(info['compromised_hosts'], (list, tuple, set)):
                            compromised_hosts.update(info['compromised_hosts'])
                        elif info['compromised_hosts'] is not None:
                             compromised_hosts.add(info['compromised_hosts'])

                except Exception as e: # Keep original error handling
                    logger.error(f"Environment step error: {e}")
                    done = True
                    next_obs = obs
                    reward = 0
                    info = {}
                    current_action_mask = None

                # Track raw reward (Keep original)
                raw_reward = reward
                episode_reward += raw_reward
                episode_steps += 1

                # Clip reward (Keep original call)
                clipped_reward = self.clip_reward(reward)

                # Track rewards for adaptive clipping (Keep original)
                self.reward_history.append(raw_reward)

                # Store transition in memory (Keep original call and arguments)
                current_state_seq = self.preprocess_observation(obs)
                self.memory.add(
                    state=current_state_seq,
                    action=action,
                    reward=clipped_reward,
                    value=value,
                    log_prob=log_prob,
                    mask=1.0 - float(done),
                    info=info
                )

                # Move to next state (Keep original)
                obs = next_obs

            # Check step limit (Keep original)
            if episode_steps >= max_steps_per_episode:
                reached_max_steps = True

            # Determine success (Keep original calls)
            is_success = self.check_success(final_info)
            if not is_success and self.target_hosts and compromised_hosts:
                is_success = bool(compromised_hosts.intersection(self.target_hosts))

            # End of episode processing (Keep original logic)
            if len(self.memory.states) > 0: # Use original condition check
                with torch.no_grad():
                    if not done:
                        final_obs_seq = self.preprocess_observation(obs)
                        self.lstm.lstm.flatten_parameters()
                        final_lstm_features, _ = self.lstm(final_obs_seq, self.hidden)
                        last_value, _, _ = self.a2c(final_lstm_features)
                        last_value = last_value.item()
                    else:
                        last_value = 0

                    # Compute returns and advantages (Keep original call)
                    self.memory.compute_returns_and_advantages(
                        last_value,
                        gamma=self.gamma,
                        gae_lambda=self.gae_lambda,
                        alpha=self.progress_advantage_alpha
                    )

                # Update policy (Keep original call)
                losses = self.update_policy()
                # Use original loss extraction logic
                policy_loss = losses['policy_loss'] if losses else 0
                value_loss = losses['value_loss'] if losses else 0
                entropy = losses['entropy'] if losses else 0
            else:
                # Keep original handling for no steps
                policy_loss = 0
                value_loss = 0
                entropy = 0
                logger.warning(f"Episode {episodes_completed+1}: No steps taken, no policy update.")


            # Compute action frequency (Keep original logic)
            if actions_taken:
                action_counts = np.bincount(actions_taken, minlength=self.action_dim)
                most_common_action = np.argmax(action_counts)
                action_frequency = action_counts[most_common_action] / len(actions_taken) if len(actions_taken) > 0 else 0.0
            else:
                most_common_action = -1
                action_frequency = 0.0


            # Update episode metrics (Keep original updates)
            episodes_completed += 1
            episode_rewards.append(episode_reward) # Keep original list if used
            self.episode_metrics['rewards'].append(episode_reward)
            self.episode_metrics['lengths'].append(episode_steps)
            self.episode_metrics['successes'].append(1 if is_success else 0)
            self.episode_metrics['entropies'].append(entropy)

            if is_success:
                self.successful_episodes += 1 # Keep original counter name

            # Log episode results (Keep original format)
            success_str = "Yes" if is_success else "No"
            timeout_str = " (Timeout)" if reached_max_steps else ""
            logger.info(f"Episode {episodes_completed}/{episodes_to_train} | "
                        f"Reward: {episode_reward:.2f} | "
                        f"Steps: {episode_steps} | "
                        f"Success: {success_str}{timeout_str} | "
                        f"Entropy: {entropy:.4f}")

            # Write to CSV log (Use the file opened at the start of the method)
            timestamp_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") # Use different var name
            # --- FIX: Reinstated try...except for robustness ---
            try:
                with open(self.episode_log_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    # Write original columns
                    writer.writerow([
                        timestamp_now,
                        scenario_name,
                        episodes_completed,
                        round(episode_reward, 2),
                        episode_steps,
                        success_str,
                        round(entropy, 4),
                        round(self.entropy_coef, 4), # Log current coef
                        most_common_action,
                        round(action_frequency, 2) # Original precision
                    ])
            except IOError as e:
                 logger.error(f"Failed to write to log file {self.episode_log_file}: {e}")
            # --- End FIX ---


            # Save best model (Keep original logic)
            if episode_reward > best_reward:
                best_reward = episode_reward
                save_path = f"{exp_dir}/best_model.pt" # Use original filename
                self.save(save_path)
                logger.info(f"New best model saved with reward: {best_reward:.2f}")

            # Periodically update reward clipping (Keep original call)
            if len(self.reward_history) >= 50: # Original threshold
                self._update_reward_clipping()

            # Create and save plots (Keep original frequency and call)
            if episodes_completed % 100 == 0 or episodes_completed == episodes_to_train:
                # Ensure plot function exists and handles potential errors
                try:
                    plot_rolling_averages(
                        scenario_name=scenario_name, # Original name argument
                        episode_metrics=self.episode_metrics,
                        metrics_dir=self.metrics_dir, # Use the correct dir for training
                        logger=logger
                    )
                except NameError:
                     logger.error("plot_rolling_averages function not found or imported.")
                except Exception as e:
                     logger.error(f"Error during plotting: {e}")


            # Log progress (Keep original frequency and format)
            if episodes_completed % 10 == 0:
                recent_episodes_count = min(10, len(self.episode_metrics['rewards']))
                if recent_episodes_count > 0:
                    avg_reward = np.mean(self.episode_metrics['rewards'][-recent_episodes_count:])
                    success_rate = np.mean(self.episode_metrics['successes'][-recent_episodes_count:])
                    logger.info(f"Progress: {episodes_completed}/{episodes_to_train} episodes | "
                                f"Recent Avg Reward: {avg_reward:.2f} | "
                                f"Recent Success Rate: {success_rate:.2%}")
                else:
                     logger.info(f"Progress: {episodes_completed}/{episodes_to_train} episodes")


        # End of training for this scenario (Keep original memory clear if it was here)
        # self.memory.clear() # Keep only if originally present here


        # Compute final statistics (Keep original calculations)
        training_time = time.time() - scenario_start_time
        avg_reward = np.mean(episode_rewards) if episode_rewards else 0.0 # Use original list if used
        success_rate = self.successful_episodes / episodes_completed if episodes_completed > 0 else 0.0

        # Save final model (Keep original path)
        final_path = f"{exp_dir}/final_model.pt" # Original filename
        self.save(final_path)

        # Create final plots (Keep original call)
        try:
            plot_rolling_averages(
                scenario_name=scenario_name, # Original name argument
                episode_metrics=self.episode_metrics,
                metrics_dir=self.metrics_dir, # Use training dir
                logger=logger
            )
        except NameError:
             logger.error("plot_rolling_averages function not found or imported.")
        except Exception as e:
             logger.error(f"Error during final plotting: {e}")


        # Log completion (Keep original format)
        logger.info(f"\nCompleted training on {scenario_name}")
        logger.info(f"Episodes: {episodes_completed}")
        logger.info(f"Training time: {training_time:.1f} seconds")
        logger.info(f"Average reward: {avg_reward:.2f}")
        logger.info(f"Success rate: {success_rate:.2%}")
        logger.info(f"Final model saved to {final_path}")

        # Record scenario performance (Keep original structure)
        scenario_performance = {
            'scenario': scenario_name,
            'completed': True,
            'episodes': episodes_completed, # Use original variable name
            'training_time': training_time,
            'avg_reward': float(avg_reward),
            'best_reward': float(best_reward),
            'successful_episodes': self.successful_episodes, # Use original counter name
            'success_rate': float(success_rate)
        }

        # Store performance (Keep original logic)
        self.scenario_performances[scenario_name] = scenario_performance
        self.scenario_history.append(scenario_name) # Keep original history tracking

        # Close environment if original code did it here
        try:
             self.env.close()
        except Exception as e:
             logger.warning(f"Error closing environment for {scenario_name}: {e}")


        return scenario_performance # Keep original return


    def train_curriculum(self, scenarios, episodes_per_scenario):
        """Train the agent through a curriculum of scenarios."""
        # Keep original logic exactly
        curriculum_dir = f"curriculum_experiments/Checkpoints" # Original path
        os.makedirs(curriculum_dir, exist_ok=True)

        config = { # Keep original config structure
            'scenarios': scenarios,
            'episodes_per_scenario': episodes_per_scenario,
            'timestamp': self.timestamp,
            'hyperparameters': {
                'hidden_size': self.lstm.hidden_dim,
                'lstm_layers': self.lstm.lstm_layers,
                # Use original LR logging if different
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'gamma': self.gamma,
                'gae_lambda': self.gae_lambda,
                'clip_ratio': self.clip_ratio,
                'value_coef': self.value_coef,
                'entropy_coef': self.entropy_coef,
                'max_grad_norm': self.max_grad_norm,
                'ppo_epochs': self.ppo_epochs,
                'ppo_batch_size': self.ppo_batch_size,
                'progress_advantage_alpha': self.progress_advantage_alpha
            }
            # Add scenario_settings if original config included it
            # 'scenario_settings': self.scenario_settings
        }

        # Keep original config saving logic
        config_path = f"{curriculum_dir}/config.json" # Use original path
        try:
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4, cls=NumpyEncoder)
        except IOError as e:
             logger.error(f"Failed to save curriculum config: {e}") # Keep original error log


        logger.info(f"\nStarting Curriculum Learning...")
        logger.info(f"Training through {len(scenarios)} scenarios")
        logger.info(f"Scenarios: {scenarios}")

        curriculum_start_time = time.time()
        all_scenario_results = []

        # Keep original curriculum loop
        for i, scenario in enumerate(scenarios):
            logger.info(f"\n\n{'#' * 80}")
            logger.info(f"CURRICULUM STEP {i + 1}/{len(scenarios)}: {scenario}")
            logger.info(f"{'#' * 80}\n")

            # Keep original episode determination logic
            if isinstance(episodes_per_scenario, dict):
                scenario_episodes = episodes_per_scenario.get(scenario, 1000)
            else:
                scenario_episodes = episodes_per_scenario


            # Keep original training call
            scenario_result = self.train_on_scenario(
                scenario_name=scenario,
                episodes_to_train=scenario_episodes
            )

            all_scenario_results.append(scenario_result)

            # Keep original intermediate results saving
            results_path = f"{curriculum_dir}/scenario_results.json" # Use original path
            try:
                with open(results_path, 'w') as f:
                    json.dump(all_scenario_results, f, indent=4, cls=NumpyEncoder)
            except IOError as e:
                 logger.error(f"Failed to save intermediate results: {e}") # Keep original error log


            # Keep original checkpoint saving
            curriculum_checkpoint = f"{curriculum_dir}/checkpoint_after_{scenario}.pt" # Original path
            self.save(curriculum_checkpoint)

            logger.info(f"Completed {i + 1}/{len(scenarios)} scenarios in curriculum")
            # Add break condition if original code had one on failure
            # if not scenario_result.get('completed', False): break


        # Keep original final model saving
        final_path = f"{curriculum_dir}/final_model.pt" # Original path
        self.save(final_path)

        # Keep original curriculum statistics calculation
        curriculum_time = time.time() - curriculum_start_time
        completed_scenarios = sum(1 for result in all_scenario_results if result.get('completed', False))
        # Add total episodes if original code calculated it
        # total_episodes = sum(r.get('episodes', 0) for r in all_scenario_results)

        curriculum_results = { # Keep original structure
            'total_scenarios': len(scenarios),
            'completed_scenarios': completed_scenarios,
            'curriculum_time': curriculum_time,
            'scenario_results': all_scenario_results,
            'final_model_path': final_path
            # Add other original fields like 'total_episodes'
        }

        # Keep original final results saving
        summary_path = f"{curriculum_dir}/curriculum_results.json" # Use original path
        try:
            with open(summary_path, 'w') as f:
                json.dump(curriculum_results, f, indent=4, cls=NumpyEncoder)
        except IOError as e:
             logger.error(f"Failed to save final curriculum results: {e}") # Keep original error log


        # Keep original final logging
        logger.info(f"\n\n{'=' * 80}")
        logger.info(f"CURRICULUM LEARNING COMPLETED")
        logger.info(f"{'=' * 80}")
        logger.info(f"Completed {completed_scenarios}/{len(scenarios)} scenarios")
        logger.info(f"Total training time: {curriculum_time / 3600:.2f} hours")
        logger.info(f"Final model saved to {final_path}")

        return curriculum_results # Keep original return


    def test(self, scenario_name, num_episodes=50, render=False, deterministic=True):
        """Test the trained agent on a specific scenario."""

        # --- Testing Logging & Metrics Setup ADDED HERE ---
        test_metrics_dir = f"metrics_Plot/Testing/{scenario_name}"
        os.makedirs(test_metrics_dir, exist_ok=True)
        test_log_file = f"{test_metrics_dir}/test_episode_logs_{scenario_name}.csv"
        try: # Add error handling for file opening (Write mode 'w')
            with open(test_log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                # Define appropriate header for test logs
                writer.writerow(['Timestamp', 'Scenario', 'Episode', 'Reward', 'Steps', 'Success', 'Timeout', 'Deterministic'])
            logger.info(f"Logging test episode data to: {test_log_file}")
        except IOError as e:
            logger.error(f"Failed to create or write header to test log file {test_log_file}: {e}")
            # Test might proceed without logging, or you could return error here
        # --- End Testing Logging Setup ---


        # Setup the environment for testing (Keep original logic)
        try:
            scenario_name_formatted = "".join(word.capitalize() for word in scenario_name.split("-"))
            render_mode = "human" if render else None
            test_env = gym.make(f"nasim:{scenario_name_formatted}PO-v0", render_mode=render_mode)

            # Keep original env reference update (if any)
            original_env = self.env # Store original if it was swapped
            self.env = test_env

            # Keep original target host fetching
            try:
                if hasattr(test_env, 'scenario') and hasattr(test_env.scenario, 'target_hosts'):
                    self.target_hosts = set(test_env.scenario.target_hosts)
                else:
                    self.target_hosts = None
            except:
                self.target_hosts = None

            # Keep original dimension checking and handling
            new_obs_dim = self._get_obs_dim(test_env)
            new_action_dim = test_env.action_space.n

            if new_obs_dim != self.obs_dim or new_action_dim != self.action_dim:
                 logger.info(f"Test environment dimensions changed. Adapting agent...")
                 old_lstm_state = self.lstm.state_dict()
                 old_a2c_state = self.a2c.state_dict()
                 self.obs_dim = new_obs_dim
                 self.action_dim = new_action_dim
                 self.lstm = LSTM(self.obs_dim, self.lstm.hidden_dim, self.lstm.lstm_layers).to(device)
                 self.a2c = A2CNetwork(self.lstm.hidden_dim, self.lstm.hidden_dim, self.action_dim).to(device)
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

            # Keep original scenario settings application (if any for test)
            # self._apply_scenario_settings(scenario_name)

            # Keep original max steps logic
            if 'tiny' in scenario_name or 'tiny-hard' in scenario_name or 'tiny-small' in scenario_name or 'small' in scenario_name or 'small-honeypot' in scenario_name or 'small-linear' in scenario_name:
                max_steps = 1000
            elif 'medium' in scenario_name or 'medium-single-site' in scenario_name or 'medium-multi-site' in scenario_name:
                max_steps = 2000
            else:
                max_steps = 5000

        except Exception as e: # Keep original error handling
            logger.error(f"Failed to create test environment: {e}")
            # Restore original env if it was swapped
            if 'original_env' in locals(): self.env = original_env
            return [] # Keep original return on error

        logger.info(f"\nTesting agent on {scenario_name} for {num_episodes} episodes")
        logger.info(f"Using deterministic actions: {deterministic}")

        # Keep original test metric lists
        test_rewards = []
        test_lengths = []
        successful_episodes = 0
        timeout_episodes = 0

        # Keep original test loop
        for episode in range(num_episodes):
            # Keep original reset logic
            observation, info = self.env.reset()
            self.obs_buffer.clear()
            self.hidden = None

            current_action_mask = info.get('action_mask', None) # Original default
            compromised_hosts = set()

            episode_reward = 0
            episode_length = 0
            actions_taken = [] # Keep original if used
            reached_max_steps = False
            final_info = {}
            done = False

            # Keep original inner test loop
            while not done and episode_length < max_steps:
                if render:
                    try: # Add try-except for render if needed
                        self.env.render()
                    except Exception as e:
                        logger.warning(f"Error rendering test environment: {e}")


                # Keep original action selection
                action, _, _ = self.select_action(observation, action_mask=current_action_mask, deterministic=deterministic)
                actions_taken.append(action) # Keep if original

                # Keep original step logic and error handling
                try:
                    step_result = self.env.step(action)
                    if len(step_result) == 5:
                        observation, reward, terminated, truncated, info = step_result
                        done = terminated or truncated
                    else:
                        observation, reward, done, info = step_result

                    current_action_mask = info.get('action_mask', None)

                    if done or episode_length == max_steps - 1:
                        final_info = info.copy() if info is not None else {}

                    if info and 'compromised_hosts' in info: # Keep original tracking
                        if isinstance(info['compromised_hosts'], (list, tuple, set)):
                            compromised_hosts.update(info['compromised_hosts'])
                        elif info['compromised_hosts'] is not None:
                             compromised_hosts.add(info['compromised_hosts'])

                except Exception as e: # Keep original error handling
                    logger.error(f"Test step error: {e}")
                    done = True
                    observation = observation
                    reward = 0
                    info = {}
                    current_action_mask = None

                # Keep original stat updates
                episode_reward += reward
                episode_length += 1

            # Keep original timeout check
            if episode_length >= max_steps:
                reached_max_steps = True
                timeout_episodes += 1

            # Keep original success check
            is_success = self.check_success(final_info)
            if not is_success and self.target_hosts and compromised_hosts:
                is_success = bool(compromised_hosts.intersection(self.target_hosts))

            # Keep original result storage
            test_rewards.append(episode_reward)
            test_lengths.append(episode_length)
            if is_success:
                successful_episodes += 1

            # Keep original action analysis if present
            # if actions_taken: ...

            # Keep original episode logging format
            timeout_str = " (Timeout)" if reached_max_steps else ""
            logger.info(f"Test Episode {episode + 1}/{num_episodes} | " +
                        f"Reward: {episode_reward:.2f} | " +
                        f"Length: {episode_length} | " +
                        f"Success: {'Yes' if is_success else 'No'}{timeout_str}")

            # --- Write to Test CSV Log ADDED HERE ---
            timestamp_now_test = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # --- FIX: Reinstated try...except for robustness ---
            try:
                with open(test_log_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        timestamp_now_test,
                        scenario_name,
                        episode + 1,
                        round(episode_reward, 2),
                        episode_length,
                        'Yes' if is_success else 'No', # Log success string
                        reached_max_steps,
                        deterministic
                    ])
            except IOError as e:
                 logger.error(f"Failed to write to test log file {test_log_file}: {e}")
            # --- End FIX ---


        # Keep original summary calculations
        avg_reward = np.mean(test_rewards) if test_rewards else 0.0
        avg_length = np.mean(test_lengths) if test_lengths else 0.0
        success_rate = successful_episodes / num_episodes if num_episodes > 0 else 0.0
        timeout_rate = timeout_episodes / num_episodes if num_episodes > 0 else 0.0

        # Keep original summary logging
        logger.info(f"\nTest Results Summary:")
        logger.info(f"Average Reward: {avg_reward:.2f}")
        logger.info(f"Average Episode Length: {avg_length:.1f}")
        logger.info(f"Success Rate: {success_rate:.2%} ({successful_episodes}/{num_episodes})")
        logger.info(f"Timeout Rate: {timeout_rate:.2%} ({timeout_episodes}/{num_episodes})")

        # Keep original environment close
        try: # Add try-except if not present
            self.env.close()
        except Exception as e:
            logger.warning(f"Error closing test environment: {e}")


        # Restore original environment if it was swapped
        if 'original_env' in locals(): self.env = original_env



        return test_rewards # Keep original return value


    def save(self, path):
        """Save the agent's networks and training state."""
        # Keep original logic exactly
        try: # Add try-except if not present
            os.makedirs(os.path.dirname(path), exist_ok=True) # Add if not present
            torch.save({
                'lstm_state_dict': self.lstm.state_dict(),
                'a2c_state_dict': self.a2c.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'obs_dim': self.obs_dim,
                'action_dim': self.action_dim,
                'hidden_dim': self.lstm.hidden_dim,
                'lstm_layers': self.lstm.lstm_layers,
                'clip_ratio': self.clip_ratio,
                'entropy_coef': self.entropy_coef,
                'reward_clip': self.reward_clip,
                'episode_count': self.episode_count,
                'scenario_history': self.scenario_history,
                'scenario_performances': self.scenario_performances,
                'ppo_epochs': self.ppo_epochs,
                'ppo_batch_size': self.ppo_batch_size,
                'progress_advantage_alpha': self.progress_advantage_alpha
                # Add timestamp if original saved it
                # 'timestamp': self.timestamp
            }, path)
        except Exception as e:
             logger.error(f"Error saving agent state to {path}: {e}", exc_info=True) # Add error log


    def load(self, path):
        """Load the agent's networks and training state, handling dimension changes."""
        # Keep original logic exactly
        if not os.path.exists(path):
            logger.error(f"Checkpoint file not found at: {path}")
            raise FileNotFoundError(f"Checkpoint file not found at: {path}")

        logger.info(f"Loading checkpoint from: {path}")
        try: # Wrap in try-except if not originally present
            checkpoint = torch.load(path, map_location=device)

            # Keep original architecture loading
            ckpt_obs_dim = checkpoint.get('obs_dim')
            ckpt_action_dim = checkpoint.get('action_dim')
            ckpt_hidden_dim = checkpoint.get('hidden_dim')
            ckpt_lstm_layers = checkpoint.get('lstm_layers')

            # Keep original validation
            if None in [ckpt_obs_dim, ckpt_action_dim, ckpt_hidden_dim, ckpt_lstm_layers]:
                logger.warning("Checkpoint missing one or more dimension/layer parameters. Using current agent defaults.")
                ckpt_obs_dim = ckpt_obs_dim if ckpt_obs_dim is not None else self.obs_dim
                ckpt_action_dim = ckpt_action_dim if ckpt_action_dim is not None else self.action_dim
                ckpt_hidden_dim = ckpt_hidden_dim if ckpt_hidden_dim is not None else self.lstm.hidden_dim
                ckpt_lstm_layers = ckpt_lstm_layers if ckpt_lstm_layers is not None else self.lstm.lstm_layers

            # Keep original network recreation logic
            recreate_networks = False
            if (ckpt_obs_dim != self.obs_dim or
                    ckpt_action_dim != self.action_dim or
                    ckpt_hidden_dim != self.lstm.hidden_dim or
                    ckpt_lstm_layers != self.lstm.lstm_layers):
                recreate_networks = True
                logger.info(
                    f"Recreating networks with dims from checkpoint: obs={ckpt_obs_dim}, action={ckpt_action_dim}, hidden={ckpt_hidden_dim}, lstm_layers={ckpt_lstm_layers}")
                self.obs_dim = ckpt_obs_dim
                self.action_dim = ckpt_action_dim

            # Keep original state loading logic
            if recreate_networks:
                self.lstm = LSTM(self.obs_dim, ckpt_hidden_dim, ckpt_lstm_layers).to(device)
                self.a2c = A2CNetwork(ckpt_hidden_dim, ckpt_hidden_dim, self.action_dim).to(device)

                # Keep original strict=False loading for recreated networks
                if 'lstm_state_dict' in checkpoint:
                    try:
                        missing_keys, unexpected_keys = self.lstm.load_state_dict(checkpoint['lstm_state_dict'], strict=False)
                        if missing_keys: logger.warning(f"LSTM loaded with missing keys: {missing_keys}")
                        if unexpected_keys: logger.warning(f"LSTM loaded with unexpected keys: {unexpected_keys}")
                        logger.info("Loaded LSTM state from checkpoint (strict=False).")
                    except Exception as e:
                        logger.error(f"Error loading LSTM state dict from checkpoint: {e}", exc_info=True)
                else:
                    logger.warning("Checkpoint missing 'lstm_state_dict'. LSTM network initialized randomly.")

                if 'a2c_state_dict' in checkpoint:
                    try:
                        missing_keys, unexpected_keys = self.a2c.load_state_dict(checkpoint['a2c_state_dict'], strict=False)
                        if missing_keys: logger.warning(f"A2C loaded with missing keys: {missing_keys}")
                        if unexpected_keys: logger.warning(f"A2C loaded with unexpected keys: {unexpected_keys}")
                        logger.info("Loaded A2C state from checkpoint (strict=False).")
                    except Exception as e:
                        logger.error(f"Error loading A2C state dict from checkpoint: {e}", exc_info=True)
                else:
                    logger.warning("Checkpoint missing 'a2c_state_dict'. A2C network initialized randomly.")

                # Keep original optimizer handling for recreated networks
                current_lr = self.optimizer.param_groups[0]['lr']
                if 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict']['param_groups']:
                     # Use original safe access if present
                     try:
                         current_lr = checkpoint['optimizer_state_dict']['param_groups'][0]['lr']
                     except (KeyError, IndexError):
                          logger.warning("Could not read LR from checkpoint optimizer state.")

                self.optimizer = optim.Adam(
                    list(self.lstm.parameters()) + list(self.a2c.parameters()),
                    lr=current_lr
                )
                logger.info(
                    f"Created new optimizer with learning rate: {current_lr}. Optimizer state not transferred due to dimension change.")

            else:
                # Keep original strict=True loading for matching dimensions
                try:
                    self.lstm.load_state_dict(checkpoint['lstm_state_dict']) # Default strict=True
                    self.a2c.load_state_dict(checkpoint['a2c_state_dict']) # Default strict=True
                    logger.info("Loaded LSTM and A2C states (strict=True).")
                except KeyError as e: # Keep original error handling
                    logger.error(f"Checkpoint missing required key: {e}. Cannot load model state.", exc_info=True)
                    raise RuntimeError(f"Checkpoint missing required key: {e}") from e
                except Exception as e: # Keep original error handling
                    logger.error(f"Error loading model state dicts: {e}", exc_info=True)
                    raise RuntimeError(f"Error loading model state dicts: {e}") from e

                # Keep original optimizer loading logic for matching dimensions
                if 'optimizer_state_dict' in checkpoint:
                    try:
                        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                        logger.info("Loaded optimizer state.")
                    except Exception as e: # Keep original handling if loading fails
                        logger.warning(f"Could not load optimizer state dict, creating new one. Error: {e}", exc_info=True)
                        current_lr = self.optimizer.param_groups[0]['lr'] # Default to current
                        if checkpoint['optimizer_state_dict']['param_groups']:
                             try:
                                 current_lr = checkpoint['optimizer_state_dict']['param_groups'][0]['lr']
                             except (KeyError, IndexError): pass # Ignore if LR not found
                        self.optimizer = optim.Adam(
                            list(self.lstm.parameters()) + list(self.a2c.parameters()),
                            lr=current_lr
                        )

            # Keep original scheduler loading logic
            if 'scheduler_state_dict' in checkpoint:
                try:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    logger.info("Loaded scheduler state.")
                except Exception as e:
                    logger.warning(f"Could not load scheduler state dict: {e}")
            # Recreate scheduler if it depends on the optimizer and optimizer was recreated
            if recreate_networks or not 'scheduler_state_dict' in checkpoint:
                 self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer, mode='max', factor=0.5, patience=20 # Use current hyperparams
                 )


            # Keep original hyperparameter loading
            self.clip_ratio = checkpoint.get('clip_ratio', self.clip_ratio)
            self.entropy_coef = checkpoint.get('entropy_coef', self.entropy_coef)
            self.reward_clip = checkpoint.get('reward_clip', self.reward_clip)
            self.ppo_epochs = checkpoint.get('ppo_epochs', self.ppo_epochs)
            self.ppo_batch_size = checkpoint.get('ppo_batch_size', self.ppo_batch_size)
            self.progress_advantage_alpha = checkpoint.get('progress_advantage_alpha', self.progress_advantage_alpha)
            # Load other hyperparameters if they were saved

            # Keep original training state loading
            self.episode_count = checkpoint.get('episode_count', 0)
            self.scenario_history = checkpoint.get('scenario_history', [])
            self.scenario_performances = checkpoint.get('scenario_performances', {})
            # Load timestamp if original did
            # self.timestamp = checkpoint.get('timestamp', self.timestamp)


            logger.info(f"Agent loaded successfully from {path}")

        except FileNotFoundError: # Keep original error handling
             raise
        except Exception as e: # Keep original error handling
            logger.error(f"Failed to load checkpoint from {path}: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load checkpoint from {path}") from e

