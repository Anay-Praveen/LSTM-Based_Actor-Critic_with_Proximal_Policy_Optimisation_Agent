import torch
import numpy as np
from utils import device, PPOTransition
import logging

logger = logging.getLogger("lstm_ppo_agent") # Get logger

class A2CMemory:
    """Relay memory buffer for Agent, used to collect episode transitions before an update."""

    def __init__(self, batch_size=64):
        self.states = [] # Stores sequences of observations [sequence_length, obs_dim]
        self.actions = []
        self.rewards = [] # Note: These are the clipped rewards used for training
        self.values = []
        self.log_probs = []
        self.masks = []  # 1 if not done, 0 if done
        self.infos = [] # Store info dictionary for progress calculation
        self.returns = []
        self.advantages = []
        self.batch_size = batch_size

        # Track previous state info to detect changes for progress calculation
        # These will be populated during the episode rollout in the agent
        self._initial_info = None


    def add(self, state, action, reward, value, log_prob, mask, info):
        """Add a transition to memory."""
        # state should be the sequence tensor [1, sequence_length, obs_dim]
        self.states.append(state.squeeze(0)) # Store as [sequence_length, obs_dim]
        self.actions.append(action)
        self.rewards.append(reward) # Store clipped reward
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.masks.append(mask)
        self.infos.append(info) # Store info dictionary


    def _calculate_progress(self, current_info, previous_info):
        """Calculate the progress term based on changes in the environment info."""
        progress = 0.0

        # Handle initial step where previous_info is None (or first info after reset)
        if previous_info is None:
             # No progress calculation possible on the very first step
             return progress
        # 1.0, if new host compromised
        # 0.5, if new vulnerability discovered
        # 0.2, if new information gathered
        # 0, otherwise

        # Note: The exact keys and structure in the NASim info dictionary might vary.
        # The following is an interpretation based on common NASim info outputs.
        # You might need to adjust this based on your specific NASim version and scenario details.

        # Use .get() with a default empty list/set to handle missing keys gracefully
        current_compromised_hosts = set(current_info.get('compromised_hosts', []))
        previous_compromised_hosts = set(previous_info.get('compromised_hosts', []))

        # Check for new host compromised (1.0)
        newly_compromised = current_compromised_hosts - previous_compromised_hosts
        if newly_compromised:
            # Add 1.0 for each newly compromised host
            progress += 1.0 * len(newly_compromised)



        # Check for new information gathered (0.2) - using scan results as proxies
        current_scanned_ports = set(current_info.get('scanned_ports', []))
        previous_scanned_ports = set(previous_info.get('scanned_ports', []))
        if len(current_scanned_ports) > len(previous_scanned_ports):
             # Check for newly scanned ports
             newly_scanned_ports = current_scanned_ports - previous_scanned_ports
             if newly_scanned_ports:
                 progress += 0.2 # 0.2, if new information gathered (via scanning ports)


        current_discovered_services = set(current_info.get('discovered_services', []))
        previous_discovered_services = set(previous_info.get('discovered_services', []))
        if len(current_discovered_services) > len(previous_discovered_services):
             # Check for newly discovered services
             newly_discovered_services = current_discovered_services - previous_discovered_services
             if newly_discovered_services:
                 progress += 0.2


        return progress


    def compute_returns_and_advantages(self, last_value, gamma=0.99, gae_lambda=0.95, alpha=1.0):
        """Compute returns and advantages using GAE with specialized progress term."""
        # Convert values to tensor and add last_value
        values = torch.tensor(self.values, dtype=torch.float, device=device)
        values = torch.cat((values, torch.tensor([last_value], dtype=torch.float, device=device)))

        advantages = []
        returns = []
        gae = 0


        all_infos = [self._initial_info] + self.infos # all_infos has size len(self.rewards) + 1


        for t in reversed(range(len(self.rewards))):
            # Calculate progress for step t (transition from state t to state t+1)
            # This progress is based on the info *after* taking action at state t (all_infos[t+1])
            # compared to the info *before* taking action at state t (all_infos[t])
            progress_term = self._calculate_progress(all_infos[t+1], all_infos[t])

            # Delta: TD error
            delta = self.rewards[t] + gamma * values[t + 1] * self.masks[t] - values[t]

            # Generalized Advantage Estimation
            gae = delta + gamma * gae_lambda * self.masks[t] * gae

            # Specialized Advantage Function: A_pen = A_GAE + alpha * Progress
            specialized_advantage = gae + alpha * progress_term

            advantages.insert(0, specialized_advantage)
            # Returns are value + advantage (using the specialized advantage)
            returns.insert(0, specialized_advantage + values[t])


        self.returns = returns
        self.advantages = advantages

    def get_batches(self):
        """Create minibatches for training.
           Yields batches of sequences for LSTM processing.
        """
        batch_size = self.batch_size
        data_size = len(self.states) # Number of transitions (steps)

        # Convert data to tensors
        # states are already tensors, stack them along a new batch dimension
        states_tensor = torch.stack(self.states).to(device) # Shape: [data_size, sequence_length, obs_dim]
        actions_tensor = torch.tensor(self.actions, dtype=torch.long).to(device) # Shape: [data_size]
        old_log_probs_tensor = torch.tensor(self.log_probs, dtype=torch.float).to(device) # Shape: [data_size]
        returns_tensor = torch.tensor(self.returns, dtype=torch.float).to(device) # Shape: [data_size]
        advantages_tensor = torch.tensor(self.advantages, dtype=torch.float).to(device) # Shape: [data_size]

        # Normalize advantages (standard practice in PPO/A2C)
        # Avoid division by zero if std is very small
        advantages_mean = advantages_tensor.mean()
        advantages_std = advantages_tensor.std()
        advantages_tensor = (advantages_tensor - advantages_mean) / (advantages_std + 1e-8)


        # Create batches
        indices = np.arange(data_size)
        np.random.shuffle(indices)

        for start in range(0, data_size, batch_size):
            end = min(start + batch_size, data_size)
            batch_idx = indices[start:end]

            # Yield batches of tensors
            yield (states_tensor[batch_idx], # Shape: [batch_size, sequence_length, obs_dim]
                   actions_tensor[batch_idx], # Shape: [batch_size]
                   old_log_probs_tensor[batch_idx], # Shape: [batch_size]
                   returns_tensor[batch_idx], # Shape: [batch_size]
                   advantages_tensor[batch_idx]) # Shape: [batch_size]

    def clear(self):
        """Clear memory after update."""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.masks.clear()
        self.infos.clear() # Clear infos
        self.returns.clear()
        self.advantages.clear()
        self._initial_info = None # Reset initial info

