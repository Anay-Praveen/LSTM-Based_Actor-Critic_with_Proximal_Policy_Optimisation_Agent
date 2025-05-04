import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import glob # To find files/directories

def plot_test_results(csv_filepath, rolling_window=20): # Added rolling_window parameter
    """
    Reads a CSV file containing test episode logs and generates smoothed plots for
    Reward vs. Episode and Steps vs. Episode using a rolling average. Saves plots
    in the same directory as the CSV file.

    Args:
        csv_filepath (str): The full path to the input CSV file.
        rolling_window (int): The window size for the rolling average calculation.
    """
    try:
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(csv_filepath)
        print(f"Successfully read CSV: {csv_filepath}")

        # --- Validate required columns ---
        required_columns = ['Episode', 'Reward', 'Steps']
        actual_columns = [col.strip() for col in df.columns] # Strip potential whitespace
        df.columns = actual_columns # Assign cleaned column names

        if not all(col in actual_columns for col in required_columns):
            missing = [col for col in required_columns if col not in actual_columns]
            print(f"Error: Missing required columns in CSV '{os.path.basename(csv_filepath)}': {missing}")
            return

        if df.empty:
             print(f"Warning: CSV file is empty: {csv_filepath}")
             return

        print(f"  Columns found: {actual_columns}")
        print(f"  Number of episodes found: {len(df)}")

        # --- Calculate Rolling Averages ---
        if len(df) >= rolling_window:
            df['Reward_Smoothed'] = df['Reward'].rolling(window=rolling_window, min_periods=1).mean()
            df['Steps_Smoothed'] = df['Steps'].rolling(window=rolling_window, min_periods=1).mean()
            plot_smoothed = True
            print(f"  Calculated rolling average with window {rolling_window}")
        else:
            # Not enough data points for the rolling window, plot raw data instead
            print(f"  Warning: Not enough data points ({len(df)}) for rolling window {rolling_window}. Plotting raw data.")
            df['Reward_Smoothed'] = df['Reward'] # Use raw data if not enough points
            df['Steps_Smoothed'] = df['Steps']
            plot_smoothed = False


        # --- Get directory for saving plots ---
        output_dir = os.path.dirname(csv_filepath)
        # Extract scenario name from the csv filename (e.g., test_episode_logs_medium -> medium)
        base_filename = os.path.splitext(os.path.basename(csv_filepath))[0]
        scenario_name_from_file = base_filename.replace("test_episode_logs_", "")


        # --- Plot 1: Smoothed Reward per Episode ---
        plt.figure(figsize=(12, 6)) # Set figure size
        # --- FIX: Corrected label ---
        reward_label = f'Smoothed Reward (Window={rolling_window})' if plot_smoothed else 'Reward per Episode'
        plt.plot(df['Episode'], df['Reward_Smoothed'], linestyle='-', label=reward_label)
        # --- End FIX ---
        # Optionally plot raw data lightly in the background
        # plt.plot(df['Episode'], df['Reward'], marker='.', linestyle='', markersize=2, color='grey', alpha=0.3, label='Raw Reward')


        plt.title(f'Test Results: Reward per Episode  ({scenario_name_from_file})')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.legend()
        plt.tight_layout() # Adjust layout

        # Define output path for reward plot (add _smoothed suffix)
        reward_plot_path = os.path.join(output_dir, f"{scenario_name_from_file}_reward_plot.png") # Use scenario name
        try:
            plt.savefig(reward_plot_path)
            print(f"  Smoothed reward plot saved to: {reward_plot_path}")
        except Exception as e:
            print(f"  Error saving smoothed reward plot: {e}")
        plt.close() # Close the plot figure to free memory

        # --- Plot 2: Smoothed Steps per Episode ---
        plt.figure(figsize=(12, 6)) # Set figure size
        # --- FIX: Corrected label ---
        steps_label = f'Smoothed Steps (Window={rolling_window})' if plot_smoothed else 'Steps per Episode'
        plt.plot(df['Episode'], df['Steps_Smoothed'], linestyle='-', color='orange', label=steps_label)
        # --- End FIX ---
        # Optionally plot raw data lightly in the background
        # plt.plot(df['Episode'], df['Steps'], marker='.', linestyle='', markersize=2, color='grey', alpha=0.3, label='Raw Steps')

        plt.title(f'Test Results: Steps per Episode  ({scenario_name_from_file})') # Use extracted name
        plt.xlabel('Episode')
        plt.ylabel('Steps Taken')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.legend()
        plt.tight_layout() # Adjust layout

        # Define output path for steps plot (add _smoothed suffix)
        steps_plot_path = os.path.join(output_dir, f"{scenario_name_from_file}_steps_plot.png") # Use scenario name
        try:
            plt.savefig(steps_plot_path)
            print(f"  Smoothed steps plot saved to: {steps_plot_path}")
        except Exception as e:
            print(f"  Error saving smoothed steps plot: {e}")
        plt.close() # Close the plot figure

    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_filepath}")
    except pd.errors.EmptyDataError:
        print(f"Error: CSV file is empty: {csv_filepath}")
    except Exception as e:
        print(f"An unexpected error occurred processing {csv_filepath}: {e}")

# --- Main Execution Logic ---
if __name__ == "__main__":
    # Define the 9 scenarios
    scenarios = [
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

    # Define the base directory where scenario folders are located


    print(f"Searching for scenario logs in  directory: ")

    found_any = False
    for scenario in scenarios:
        print(f"\nProcessing scenario: {scenario}")

        # Construct the expected directory pattern (allows for optional timestamp)
        # We look for directories that START with the scenario name
        scenario_dir_pattern = os.path.join(f"{scenario}*")
        matching_dirs = glob.glob(scenario_dir_pattern)

        if not matching_dirs:
            print(f"  No directory found matching pattern: {scenario_dir_pattern}")
            continue

        # If multiple directories match (e.g., different timestamps), process the first one.
        target_dir = matching_dirs[0]
        print(f"  Found directory: {target_dir}")

        # Construct the expected CSV file path within that directory
        csv_filename = f"test_episode_logs_{scenario}.csv"
        csv_path = os.path.join(target_dir, csv_filename)

        # Check if the CSV file exists and plot
        if os.path.exists(csv_path):
            print(f"  Found CSV file: {csv_path}")
            plot_test_results(csv_path, rolling_window=20) # Specify the window size here
            found_any = True
        else:
            print(f"  CSV file not found: {csv_path}")

    if not found_any:
        print("\nNo test log CSV files were found for any scenario in the expected locations.")
