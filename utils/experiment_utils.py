import os
from typing import Any
from omegaconf import OmegaConf
import glob
import json
import logging
import torch
from typing import Dict, List, Optional, Tuple, Union

from utils.hash import hash_config

logger = logging.getLogger(__name__)


def get_experiment_info(experiment_dir: str) -> Dict[str, Any]:
    """
    Get information about an experiment.
    
    Args:
        experiment_dir: The directory of the experiment
        
    Returns:
        A dictionary with experiment information
    """
    if not os.path.exists(experiment_dir):
        raise ValueError(f"Experiment directory {experiment_dir} does not exist")
    
    result = {
        'name': os.path.basename(experiment_dir),
        'dir': experiment_dir,
    }
    
    # Load configuration
    config_path = os.path.join(experiment_dir, 'config.yaml')
    if os.path.exists(config_path):
        try:
            config = OmegaConf.load(config_path)
            result['config'] = config
            result['config_hash'] = hash_config(config)
            
        except Exception as e:
            logger.error(f"Error loading configuration from {config_path}: {e}")
            result['config'] = None
            result['config_hash'] = None
    else:
        logger.warning(f"No config.yaml found in {experiment_dir}")
        result['config'] = None
        result['config_hash'] = None
    
    
    return result