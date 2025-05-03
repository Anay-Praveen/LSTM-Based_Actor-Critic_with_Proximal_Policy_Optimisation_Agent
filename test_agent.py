# agent_test.py (Interactive Version)
import gymnasium as gym
import nasim
import torch
import os
import datetime # Used indirectly by logger likely
import time # Used in agent.test for rendering delay
import sys # For exiting

# --- Import the Agent and Utilities ---
# Make sure agent.py, utils.py, networks.py, memory.py,
# and plot.py are in the same directory or accessible in the Python path.
try:
    # Updated import to use agent.py
    from agent import NASIMOffensiveAgent
    from utils import logger, device # Assuming logger and device are defined in utils
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    # Updated error message
    print("Please ensure agent.py, utils.py, networks.py, memory.py, and plot.py are accessible.")
    sys.exit(1)

# --- Define Available Scenarios ---
AVAILABLE_SCENARIOS = [
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

# --- Helper Functions for User Input ---
def get_validated_input(prompt, valid_options=None, input_type=str):
    """Gets and validates user input."""
    while True:
        try:
            user_input = input(prompt).strip()
            value = input_type(user_input)
            if valid_options:
                if isinstance(valid_options, range) and value not in valid_options:
                     print(f"Invalid choice. Please enter a number between {valid_options.start} and {valid_options.stop - 1}.")
                elif not isinstance(valid_options, range) and value not in valid_options:
                     print(f"Invalid choice. Please enter one of: {valid_options}")
                else:
                    return value # Valid input
            else:
                 return value # No specific options to check against
        except ValueError:
            print(f"Invalid input. Please enter a valid {input_type.__name__}.")
        except EOFError:
             print("\nOperation cancelled by user.")
             sys.exit(0)
        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
            sys.exit(0)


def get_yes_no_input(prompt):
    """Gets a 'yes' or 'no' input, returns True for yes, False for no."""
    while True:
        response = input(prompt + " (yes/no): ").strip().lower()
        if response in ['yes', 'y']:
            return True
        elif response in ['no', 'n']:
            return False
        else:
            print("Invalid input. Please enter 'yes' or 'no'.")


# --- Main Testing Logic ---
def run_test(agent, scenario_name, num_episodes, render_env, deterministic_actions):
    """Runs the agent's test method for a given scenario."""
    logger.info(f"\n--- Testing Scenario: {scenario_name} ---")
    try:
        test_rewards = agent.test(
            scenario_name=scenario_name,
            num_episodes=num_episodes,
            render=render_env,
            deterministic=deterministic_actions # Pass the determined value
        )
        logger.info(f"Testing completed for {scenario_name}.")
        if test_rewards:
             avg_reward = sum(test_rewards) / len(test_rewards)
             logger.info(f"Average reward over {len(test_rewards)} episodes: {avg_reward:.3f}")
        else:
            logger.warning("No rewards recorded, testing might have failed or skipped episodes.")

    except Exception as e:
        logger.error(f"An error occurred during testing on scenario '{scenario_name}': {e}")
        import traceback
        logger.error(traceback.format_exc()) # Log full traceback


def main():
    """
    Main function to load and test the NASIM Offensive Agent interactively.
    """
    print("--- NASIM Offensive Agent Interactive Test ---")

    # 1. Choose Test Type
    print("\nSelect Testing Mode:")
    print("  1: Curriculum-based Testing (test one model on all scenarios)")
    print("  2: Individual Scenario Testing")
    test_mode = get_validated_input("Enter choice (1 or 2): ", valid_options=[1, 2], input_type=int)

    # 2. Get Common Parameters
    print("\nEnter testing parameters:")
    model_path = ""
    while not model_path or not os.path.exists(model_path):
         model_path_input = input("Enter path to the saved agent model file (.pt): ").strip()
         if os.path.exists(model_path_input):
             model_path = model_path_input
         else:
             print(f"Error: Model file not found at '{model_path_input}'. Please try again.")

    num_episodes = get_validated_input("Enter number of episodes per scenario: ", input_type=int)
    while num_episodes <= 0:
         print("Number of episodes must be positive.")
         num_episodes = get_validated_input("Enter number of episodes per scenario: ", input_type=int)

    render_env = get_yes_no_input("Render the environment during testing?")

    # Deterministic actions are now always False (sampling actions)
    deterministic_actions = False
    logger.info("Deterministic actions set to False (sampling actions).") # Log the setting

    # 3. Instantiate Agent and Load Model
    logger.info(f"\nLoading model from: {model_path}")
    logger.info(f"Using Device: {device}")

    try:
        # Create dummy env for initialization using a known simple scenario
        dummy_env = gym.make("nasim:TinyPO-v0")
        # Initialize the agent using the dummy environment
        agent = NASIMOffensiveAgent(env=dummy_env)
        # Load the saved model state
        agent.load(model_path)
        # Close the dummy environment as it's no longer needed
        dummy_env.close()
        logger.info("Agent instantiated and model loaded successfully.")
    except gym.error.Error as e:
        logger.error(f"Failed to create dummy Gym environment for agent initialization: {e}")
        return
    except FileNotFoundError:
        logger.error(f"Model file not found during load attempt: {model_path}")
        return
    except Exception as e:
        logger.error(f"Failed to instantiate or load agent: {e}")
        import traceback
        logger.error(traceback.format_exc()) # Log full traceback for debugging
        return

    # 4. Execute Chosen Test Mode
    if test_mode == 1:
        # Curriculum-based Testing
        logger.info("\n--- Starting Curriculum Test ---")
        logger.info(f"Testing model '{model_path}' on all {len(AVAILABLE_SCENARIOS)} scenarios.")
        for scenario in AVAILABLE_SCENARIOS:
            # Pass the hardcoded 'deterministic_actions' value
            run_test(agent, scenario, num_episodes, render_env, deterministic_actions)
            time.sleep(1) # Small pause between scenarios
        logger.info("\n--- Curriculum Test Finished ---")

    elif test_mode == 2:
        # Individual Scenario Testing
        logger.info("\n--- Starting Individual Scenario Test ---")
        while True:
            print("\nAvailable Scenarios:")
            for i, name in enumerate(AVAILABLE_SCENARIOS):
                print(f"  {i + 1}: {name}")
            print("  0: Exit")

            prompt = f"Enter scenario number to test (1-{len(AVAILABLE_SCENARIOS)}) or 0 to exit: "
            scenario_choice = get_validated_input(prompt, valid_options=range(len(AVAILABLE_SCENARIOS) + 1), input_type=int)

            if scenario_choice == 0:
                logger.info("Exiting individual testing.")
                break
            else:
                selected_scenario = AVAILABLE_SCENARIOS[scenario_choice - 1]
                # Pass the hardcoded 'deterministic_actions' value
                run_test(agent, selected_scenario, num_episodes, render_env, deterministic_actions)

            # Ask to continue
            if not get_yes_no_input("\nTest another individual scenario?"):
                break
        logger.info("\n--- Individual Scenario Test Finished ---")

    print("\n--- NASIM Offensive Agent Interactive Test Complete ---")


# --- Run the main function directly when the script is executed ---
if __name__ == "__main__":
    # Basic logger setup if not already configured by utils.py
    # This ensures logger messages are displayed.
    # If utils.py already configures logging, this might be redundant
    # but generally safe.
    import logging
    # Check if the specific logger we use ('lstm_ppo_agent' from utils) has handlers
    # If not, set up a basic configuration.
    agent_logger = logging.getLogger("lstm_ppo_agent") # Get the specific logger instance
    if not agent_logger.hasHandlers():
         # Configure the specific logger if it hasn't been configured yet
         # This avoids interfering with potential root logger configurations.
         handler = logging.StreamHandler()
         formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
         handler.setFormatter(formatter)
         agent_logger.addHandler(handler)
         agent_logger.setLevel(logging.INFO)
         # Prevent messages from propagating to the root logger if it has handlers
         agent_logger.propagate = False


    main()
