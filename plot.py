import matplotlib.pyplot as plt
import numpy as np
import logging # Import logging

# Note: The logger instance needs to be passed to plot_rolling_averages

def calculate_rolling_average(data, window_size=100):
    """Calculate rolling average for a given data list."""
    if len(data) < window_size:
        return []
    # Ensure window_size does not exceed data length
    actual_window_size = min(window_size, len(data))
    if actual_window_size == 0:
         return []
    return [sum(data[i:i + actual_window_size]) / actual_window_size for i in range(len(data) - actual_window_size + 1)]

def plot_rolling_averages(scenario_name, episode_metrics, metrics_dir, logger, window_size=100):
    """Plot rolling averages for rewards, steps, and successes per episode during training."""
    # Calculate rolling averages using the helper function
    rolling_rewards = calculate_rolling_average(episode_metrics['rewards'], window_size)
    rolling_steps = calculate_rolling_average(episode_metrics['lengths'], window_size)
    rolling_successes = calculate_rolling_average(episode_metrics['successes'], window_size)

    if not rolling_rewards and not rolling_steps and not rolling_successes:
        logger.warning(f"Not enough data to calculate rolling averages for {scenario_name} with window {window_size}")
        return

    # Determine the number of episodes to plot based on available rolling data
    num_episodes_to_plot = max(len(rolling_rewards), len(rolling_steps), len(rolling_successes))
    if num_episodes_to_plot == 0:
         logger.warning(f"No rolling average data to plot for {scenario_name}.")
         return

    # Calculate the correct episode range for the x-axis
    # Rolling average starts after 'window_size - 1' episodes.
    # The x-axis should represent the *end* episode of each window.
    start_episode_index = window_size - 1
    end_episode_index = len(episode_metrics['rewards']) - 1
    # +1 because range is exclusive at the end, and we want episode numbers (1-based)
    episodes_range = range(start_episode_index + 1, end_episode_index + 2)


    # Plot rewards with updated style
    plt.figure(figsize=(10, 6)) # Adjusted figure size
    if rolling_rewards:
         # Ensure the length matches the calculated episodes_range
         plt.plot(episodes_range, rolling_rewards, label=f'Rolling Avg Reward (Window={window_size})', color='darkblue', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Rolling Average Reward')
    plt.title(f'Rolling Average Reward / Episodes - {scenario_name}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout() # Adjust layout
    plt.savefig(f"{metrics_dir}/{scenario_name}_rolling_rewards.png")
    plt.close()
    logger.info(f"Saved rolling average reward plot for {scenario_name}")


    # Plot steps with updated style
    plt.figure(figsize=(10, 6)) # Adjusted figure size
    if rolling_steps:
         # Ensure the length matches the calculated episodes_range
         plt.plot(episodes_range, rolling_steps, label=f'Rolling Avg Steps (Window={window_size})', color='darkorange', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Rolling Average Steps')
    plt.title(f'Rolling Average Steps / Episodes - {scenario_name}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout() # Adjust layout
    plt.savefig(f"{metrics_dir}/{scenario_name}_rolling_steps.png")
    plt.close()
    logger.info(f"Saved rolling average steps plot for {scenario_name}")

    # Plot rolling success rate with updated style
    plt.figure(figsize=(10, 6)) # Adjusted figure size
    if rolling_successes:
         # Ensure the length matches the calculated episodes_range
         plt.plot(episodes_range, rolling_successes, label=f'Rolling Avg Success Rate (Window={window_size})', color='darkgreen', linewidth=2) # Changed color
    plt.xlabel('Episode')
    plt.ylabel('Rolling Average Success Rate')
    plt.title(f'Rolling Average Success Rate / Episodes - {scenario_name}')
    plt.ylim(0, 1.1) # Set y-limit for success rate (0 to 1 or slightly above)
    plt.legend()
    plt.grid(True)
    plt.tight_layout() # Adjust layout
    plt.savefig(f"{metrics_dir}/{scenario_name}_rolling_success_rate.png")
    plt.close()
    logger.info(f"Saved rolling average success rate plot for {scenario_name}")

