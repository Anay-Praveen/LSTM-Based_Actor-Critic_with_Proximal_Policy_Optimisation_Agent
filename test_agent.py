import gymnasium as gym
import os
import time
import sys
import logging
import traceback # Import traceback for detailed error logging

# --- Import the Agent and Utilities ---
try:
    # Ensure agent.py and other necessary files are importable
    from agent import NASIMOffensiveAgent
    # Make sure logger and device are correctly defined/imported in your utils.py
    from utils import logger, device
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    print("Please ensure agent.py, utils.py, networks.py, memory.py, and plot.py are accessible.")
    sys.exit(1)
except NameError as e:
    # Handle cases where logger or device might not be defined in utils
    print(f"Error importing from utils: {e}. Ensure logger and device are defined.")
    # Set up a basic logger if utils.logger is missing
    logger = logging.getLogger("test_agent_fallback")
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    # Set device fallback if utils.device is missing
    try:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.warning("utils.device not found, falling back to auto-detected device.")
    except ImportError:
        logger.error("PyTorch not found. Cannot set device.")
        sys.exit(1)


# --- Define Available Scenarios ---
# Ensure this list matches the scenarios your models were trained on
# and the directory names within 'scenario_checkpoints'
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

# --- Base directory where scenario-specific models are saved ---
SCENARIO_CHECKPOINT_DIR = "scenario_checkpoints"

# --- Fixed Testing Parameters ---
NUM_EPISODES_PER_SCENARIO = 100
RENDER_ENVIRONMENT = False
DETERMINISTIC_ACTIONS = False # False = sampling actions


# --- Helper Functions for User Input (Simplified) ---
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


# --- Main Testing Logic Function (Takes agent as input) ---
def run_test(agent, scenario_name, num_episodes, render_env, deterministic_actions):
    """Runs the agent's test method for a given scenario."""
    logger.info(f"\n--- Testing Scenario: {scenario_name} ---")
    try:
        test_rewards = agent.test(
            scenario_name=scenario_name,
            num_episodes=num_episodes,
            render=render_env,
            deterministic=deterministic_actions
        )
        if test_rewards:
             avg_reward = sum(test_rewards) / len(test_rewards)
             logger.info(f"Testing completed for {scenario_name}. Average reward over {len(test_rewards)} episodes: {avg_reward:.3f}")
             return avg_reward
        else:
            logger.warning(f"No rewards recorded for {scenario_name}, testing might have failed or skipped episodes.")
            return None

    except Exception as e:
        logger.error(f"An error occurred during testing on scenario '{scenario_name}': {e}")
        logger.error(traceback.format_exc())
        return None


def main():
    """
    Main function to test NASIM Offensive Agents by loading scenario-specific models,
    with fixed episode count and rendering settings.
    """
    print("--- NASIM Offensive Agent Interactive Test (Scenario-Specific Models) ---")
    print(f" Episodes per scenario fixed to: {NUM_EPISODES_PER_SCENARIO}")
    print(f" Environment rendering fixed to: {'Yes' if RENDER_ENVIRONMENT else 'No'}")
    print(f" Action selection fixed to: {'Deterministic' if DETERMINISTIC_ACTIONS else 'Sampling'}")


    # 1. Choose Test Type
    print("\nSelect Testing Mode:")
    print(f"  1: Curriculum-based Testing ({len(AVAILABLE_SCENARIOS)} scenarios sequentially)")
    print("  2: Individual Scenario Testing")
    test_mode = get_validated_input("Enter choice (1 or 2): ", valid_options=[1, 2], input_type=int)

    # --- Dummy Environment for Agent Initialization ---
    try:
        logger.info("Creating dummy environment for agent initialization...")
        dummy_env = gym.make("nasim:TinyPO-v0")
        logger.info("Dummy environment created.")
    except Exception as e:
        logger.error(f"Failed to create dummy Gym environment for agent initialization: {e}")
        logger.error(traceback.format_exc())
        return

    # --- Execute Chosen Test Mode ---
    all_results = {}

    if test_mode == 1:
        # Curriculum-based Testing (Sequential, Scenario-Specific Models)
        logger.info("\n--- Starting Sequential Curriculum Test (Scenario-Specific Models) ---")
        logger.info(f"Testing sequentially on all {len(AVAILABLE_SCENARIOS)} scenarios.")
        logger.info(f"Looking for models in base directory: '{SCENARIO_CHECKPOINT_DIR}'")

        for scenario in AVAILABLE_SCENARIOS:
            logger.info(f"\n{'='*20} Preparing Scenario: {scenario} {'='*20}")
            scenario_model_path = os.path.join(SCENARIO_CHECKPOINT_DIR, scenario, "final_model.pt")

            if not os.path.exists(scenario_model_path):
                logger.warning(f"Model file not found for scenario '{scenario}' at '{scenario_model_path}'. Skipping.")
                all_results[scenario] = None
                continue

            try:
                # Instantiate a FRESH agent instance for each scenario
                logger.info(f"Instantiating agent using dummy env...")
                agent = NASIMOffensiveAgent(env=dummy_env) # Initialize with dummy

                # Load the SCENARIO-SPECIFIC model
                logger.info(f"Loading model for '{scenario}' from: {scenario_model_path}")
                logger.info(f"Using Device: {device}")
                agent.load(scenario_model_path)
                logger.info(f"Model for '{scenario}' loaded successfully.")

                # Run the test for this scenario with the loaded agent
                avg_reward = run_test(
                    agent,
                    scenario,
                    NUM_EPISODES_PER_SCENARIO, # Use fixed value
                    RENDER_ENVIRONMENT,        # Use fixed value
                    DETERMINISTIC_ACTIONS      # Use fixed value
                )
                all_results[scenario] = avg_reward

            except FileNotFoundError:
                logger.error(f"Model file not found during load attempt: {scenario_model_path}")
                all_results[scenario] = None
            except Exception as e:
                logger.error(f"Failed to instantiate, load, or test agent for scenario '{scenario}': {e}")
                logger.error(traceback.format_exc())
                all_results[scenario] = None

            time.sleep(1) # Small pause between scenarios

        logger.info("\n--- Sequential Curriculum Test Finished ---")
        logger.info("Average Rewards per Scenario:")
        for scenario, avg_reward in all_results.items():
             result_str = f"{avg_reward:.2f}" if avg_reward is not None else "Skipped/Error"
             logger.info(f"  {scenario}: {result_str}")


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

            selected_scenario = AVAILABLE_SCENARIOS[scenario_choice - 1]
            logger.info(f"\n{'='*20} Preparing Scenario: {selected_scenario} {'='*20}")
            scenario_model_path = os.path.join(SCENARIO_CHECKPOINT_DIR, selected_scenario, "final_model.pt")

            if not os.path.exists(scenario_model_path):
                logger.warning(f"Model file not found for scenario '{selected_scenario}' at '{scenario_model_path}'. Cannot test.")
            else:
                try:
                    # Instantiate a FRESH agent instance
                    logger.info(f"Instantiating agent using dummy env...")
                    agent = NASIMOffensiveAgent(env=dummy_env) # Initialize with dummy

                    # Load the SCENARIO-SPECIFIC model
                    logger.info(f"Loading model for '{selected_scenario}' from: {scenario_model_path}")
                    logger.info(f"Using Device: {device}")
                    agent.load(scenario_model_path)
                    logger.info(f"Model for '{selected_scenario}' loaded successfully.")

                    # Run the test
                    run_test(
                        agent,
                        selected_scenario,
                        NUM_EPISODES_PER_SCENARIO, # Use fixed value
                        RENDER_ENVIRONMENT,        # Use fixed value
                        DETERMINISTIC_ACTIONS      # Use fixed value
                    )

                except FileNotFoundError:
                    logger.error(f"Model file not found during load attempt: {scenario_model_path}")
                except Exception as e:
                    logger.error(f"Failed to instantiate, load, or test agent for scenario '{selected_scenario}': {e}")
                    logger.error(traceback.format_exc())

            # Ask to continue - simplifying this slightly
            continue_testing = get_validated_input("\nTest another individual scenario? (yes/no): ", valid_options=['yes', 'y', 'no', 'n'])
            if continue_testing.lower() in ['no', 'n']:
                 break
        logger.info("\n--- Individual Scenario Test Finished ---")

    # Clean up dummy environment
    try:
        dummy_env.close()
        logger.info("Dummy environment closed.")
    except Exception as e:
        logger.warning(f"Could not close dummy environment: {e}")


    print("\n--- NASIM Offensive Agent Interactive Test Complete ---")


# --- Run the main function directly when the script is executed ---
if __name__ == "__main__":
    # Basic logger setup if not handled by utils.py
    agent_logger = logging.getLogger("lstm_ppo_agent") # Match logger name used in agent.py
    if not agent_logger.hasHandlers():
         handler = logging.StreamHandler()
         formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
         handler.setFormatter(formatter)
         agent_logger.addHandler(handler)
         agent_logger.setLevel(logging.INFO)
         agent_logger.propagate = False

         test_script_logger = logging.getLogger("test_agent_fallback")
         if not test_script_logger.hasHandlers():
              test_script_logger.addHandler(handler)
              test_script_logger.setLevel(logging.INFO)
              test_script_logger.propagate = False

    main()