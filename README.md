# LSTM-Based Actor-Critic Neural Network with Proximal Policy Optimisation Agent for Automated Network Penetration Testing

This repository contains the implementation of a reinforcement learning agent designed for automated network penetration testing within the NASim (Network Attack Simulator) environment. The agent utilizes a Long Short-Term Memory (LSTM) network combined with an Advantage Actor-Critic (A2C) framework and Proximal Policy Optimisation (PPO)-style updates to navigate complex, partially observable network scenarios and identify vulnerabilities.

## Abstract

The escalating complexity and dynamic nature of contemporary cyber threats present significant challenges to established network security practices. Conventional penetration testing methodologies, often reliant on time-consuming manual processes, struggle to provide the continuous and comprehensive security validation required for modern network infrastructures. This project addresses this need by developing a novel reinforcement learning agent employing an LSTM network to handle sequential observations and partial observability. Policy optimisation is achieved using stable PPO-style updates within an A2C framework. A structured curriculum learning approach facilitates progressive skill acquisition across simulated network scenarios of increasing difficulty in NASim, aiming to create an autonomous agent capable of intelligently identifying vulnerabilities with significantly reduced human intervention.

*(Adapted from the provided report)*

## Features

* **Reinforcement Learning Agent:** Implements an A2C agent optimized with PPO-style clipped objectives and GAE.
* **LSTM for Sequential Processing:** Utilizes an LSTM network to handle partial observability and learn from sequences of observations, crucial for multi-step attacks.
* **Curriculum Learning:** Trains the agent progressively across a series of NASim scenarios with increasing complexity (`tiny` -> `small` -> `medium` variants).
* **Knowledge Transfer:** Implements mechanisms to transfer learned weights between scenarios with different observation/action space dimensions.
* **Specialized Advantage Function:** Incorporates a progress-based term into the GAE calculation to provide denser rewards in sparse-reward penetration testing environments.
* **Adaptive Hyperparameters:** Adjusts learning rate, entropy coefficient, and clipping ratio based on the current training scenario. Dynamically adapts entropy coefficient based on performance.
* **NASim Integration:** Designed to work seamlessly with the Network Attack Simulator (NASim) environment.
* **Performance Logging & Plotting:** Logs detailed episode metrics (rewards, steps, success, entropy) to CSV and generates rolling average plots for visualization.
* **Interactive Testing:** Includes a separate script (`test_agent.py`) for interactively testing trained models on specific scenarios or the full curriculum.

## Methodology Overview

1.  **Observation Processing:** Raw observations from NASim are flattened, normalized, and fed into a sequence buffer.
2.  **LSTM Network:** Processes the sequence of observations to generate a context-aware state representation, handling temporal dependencies and partial observability.
3.  **A2C Network:**
    * **Actor:** Outputs a policy (probability distribution over actions) based on the LSTM state.
    * **Critic:** Estimates the value (expected future return) of the current state.
4.  **Action Selection:** An action is sampled from the actor's policy distribution, potentially using action masking provided by NASim.
5.  **Advantage Estimation:** Generalized Advantage Estimation (GAE) is calculated, incorporating a specialized progress term (`alpha * Progress`) rewarding intermediate steps like host compromises or information discovery.
6.  **PPO-Style Update:** The actor and critic networks are updated using the collected transitions. The policy update uses the PPO clipped surrogate objective function for stability. Value loss and an entropy bonus (with adaptive coefficient) are included in the total loss.
7.  **Curriculum Learning:** The agent is trained sequentially on scenarios (`tiny` -> `small` -> `medium`), transferring weights to adapt to changing environment dimensions.

## Requirements

* Python 3.x
* PyTorch (`torch`)
* Gymnasium (`gymnasium`)
* NASim (`nasim`)
* NumPy (`numpy`)
* Matplotlib (`matplotlib`)

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-directory>
    ```
2.  **Install dependencies:**
    * It's recommended to use a virtual environment:
        ```bash
        python -m venv venv
        source venv/bin/activate # On Windows use `venv\Scripts\activate`
        ```
    * Install NASim (follow instructions from the official NASim repository/documentation).
    * Install other requirements:
        ```bash
        pip install torch gymnasium numpy matplotlib
        ```

## Usage

### Training

1.  Run the main training script. This will execute the curriculum learning process sequentially through the predefined scenarios (`tiny` -> `small` -> `medium`).
    ```bash
    python main.py
    ```
2.  Training progress, logs, and plots will be saved in directories like `scenario_checkpoints/`, `curriculum_experiments/`, and `metrics_Plot/`.
3.  Checkpoints (`best_model.pt`, `final_model.pt`) will be saved for each scenario and for the overall curriculum.

### Testing

1.  Use the interactive testing script to evaluate a trained model.
    ```bash
    python test_agent.py
    ```
2.  The script will prompt you to:
    * Select the testing mode (Curriculum-based or Individual Scenario).
    * Enter the path to the saved model file (`.pt`).
    * Specify the number of episodes per scenario.
    * Choose whether to render the environment.
    * If testing individual scenarios, select the desired scenario from the list.
3.  Test results (average reward, steps, success rate) will be logged to the console and `experiment.log`.

## File Structure
.
*├── agent.py              # Core RL logic
*├── main.py               # Script for curriculum training
*├── memory.py             # Memory buffer for storing transitions
*├── networks.py           # LSTM and A2C network architectures
*├── plot.py               # Utility functions for plotting metrics
*├── test_agent.py         # Script for interactive testing
*├── utils.py              # Utility functions for logging, seeds, etc.
*├── scenario_checkpoints/ # Directory for scenario model checkpoints
*├── curriculum_experiments/ # Directory for curriculum training results
*├── metrics_Plot/         # Directory for training performance plots
*├── experiment.log        # Log file for training/testing output
*└── README.md             # This file

## Results Overview

The agent demonstrates successful learning across the NASim curriculum, achieving high success rates and significantly outperforming benchmark results in terms of efficiency (fewer steps) and overall reward, especially in complex `medium` scenarios with varied topologies. Detailed results, performance curves, and comparisons can be found in the generated plots (`metrics_Plot/`) and the accompanying research report (`final_report__20509910_ (4).pdf`). Network topology was found to be a critical factor influencing task difficulty.

## Acknowledgements

*(Refer to the Acknowledgement section in `final_report__20509910_ (4).pdf`)*

## References

*(Refer to the References section in `final_report__20509910_ (4).pdf`)*
