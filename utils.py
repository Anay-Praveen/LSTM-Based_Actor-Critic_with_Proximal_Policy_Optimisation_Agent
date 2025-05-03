import json
import numpy as np
from collections import namedtuple
import logging
import torch
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("experiment.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("lstm_ppo_agent")

class NumpyEncoder(json.JSONEncoder):
    """Custom JSONEncoder for handling NumPy data types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# Transition storage for PPO
PPOTransition = namedtuple('PPOTransition',
                           ['state', 'action', 'reward',
                            'log_prob', 'value', 'mask', 'info'])

# Set up CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

# Set seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)

