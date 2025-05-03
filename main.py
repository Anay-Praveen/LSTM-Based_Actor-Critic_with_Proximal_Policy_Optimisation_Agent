#!/usr/bin/env python
"""
Optimized LSTM-A2C Agent for Network Penetration Testing

Features:
- Episode-based curriculum learning across multiple network environments
- Efficient neural network architecture with LSTM for sequential decision making
- Adaptive entropy coefficient for balanced exploration/exploitation
- Knowledge transfer between scenarios with different dimensions
- Success tracking based on target host compromise
"""

import gymnasium as gym
# Import the agent module
from agent import NASIMOffensiveAgent # Assuming agent.py is in the same directory
# Import utils for device and logger if needed in main
from utils import device, logger # Assuming utils.py is in the same directory


def main():
    """Main function to run the agent."""
    # Define the curriculum of scenarios
    curriculum_scenarios = [
        'tiny',
        'tiny-hard',
        'tiny-small',
        'small',
        'small-honeypot',
        'small-linear',
        'medium',
        'medium-single-site',
        'medium-multi-site',
    ]

    # Create initial environment to initialize agent
    # This initial environment is used to get the observation/action space dimensions
    # before the curriculum starts. The agent will create new environments
    # for each scenario in train_curriculum.
    try:
        # Convert scenario name to NASim format
        initial_scenario_formatted = "".join(word.capitalize() for word in curriculum_scenarios[0].split("-"))
        initial_env = gym.make(f"nasim:{initial_scenario_formatted}PO-v0")
    except Exception as e:
        logger.error(f"Failed to create initial environment {curriculum_scenarios[0]}: {e}")
        return # Exit if initial environment creation fails


    # Define episodes per scenario as requested
    episodes_per_scenario = {}
    for scenario in curriculum_scenarios:
        # Set environment-specific episode limits
        # Keeping original episode limits
        if 'tiny' in scenario or 'tiny-hard' in scenario or 'tiny-small' in scenario or 'small' in scenario or 'small-honeypot' in scenario or 'small-linear' in scenario:
            episodes_per_scenario[scenario] = 500
        elif 'medium' in scenario or 'medium-single-site' in scenario or 'medium-multi-site' in scenario:
            episodes_per_scenario[scenario] = 2000
        else:
            episodes_per_scenario[scenario] = 2000  # Default

    # Initialize agent
    # Pass the initial_env to the agent constructor
    agent = NASIMOffensiveAgent(
        env=initial_env, # Pass the initial environment
        hidden_size=128,
        lstm_layers=5,
        learning_rate=0.0003,
        gamma=0.99,
        gae_lambda=0.95, # Keeping GAE
        clip_ratio=0.2, # Keeping clip_ratio
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        ppo_epochs=4, # Keeping original name
        ppo_batch_size=64, # Keeping original name
        progress_advantage_alpha=1.0 # Default alpha value, can be overridden by scenario settings
    )

    # Close the initial environment after agent initialization
    initial_env.close()

    # Train through curriculum sequentially
    logger.info("Starting curriculum training...")
    curriculum_results = agent.train_curriculum(
        scenarios=curriculum_scenarios,
        episodes_per_scenario=episodes_per_scenario
    )
    logger.info("Curriculum training finished.")
    logger.info(f"Curriculum Results: {curriculum_results}")

    # Testing logic has been moved to a separate file (test_agent.py)
    # You can now run test_agent.py to evaluate the trained model.


if __name__ == "__main__":
    main()

