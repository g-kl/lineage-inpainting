"""Config hashing and output directory management."""

import hashlib
import json
import os
import logging
from typing import Any, Union

import yaml
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

DEFAULT_EXCLUDE_KEYS = ["training", "hydra", "device"]


def hash_config(
    config: Union[dict[str, Any], DictConfig],
    exclude_keys: list[str] | None = None,
) -> str:
    """Generate a deterministic MD5 hash from a configuration.

    Parameters
    ----------
    config : dict or DictConfig
        Full configuration to hash.
    exclude_keys : list of str, optional
        Top-level keys to exclude.  Defaults to training, hydra, and
        device.

    Returns
    -------
    str
        Hex digest of the config hash.
    """
    if isinstance(config, DictConfig):
        config_dict = OmegaConf.to_container(config, resolve=True)
    else:
        config_dict = config.copy()

    exclude = set(exclude_keys or DEFAULT_EXCLUDE_KEYS)
    filtered = {k: v for k, v in config_dict.items() if k not in exclude}

    sorted_json = json.dumps(filtered, sort_keys=True)
    return hashlib.md5(sorted_json.encode()).hexdigest()


def get_output_dir(
    config: Union[dict[str, Any], DictConfig],
    base_dir: str = "./outputs",
    experiment_name: str | None = None,
    create: bool = True,
) -> str:
    """Get a deterministic output directory based on config hash.

    Returns ``{base_dir}/{experiment_name}_{hash}/``.  Creates the
    directory and saves the resolved config on first call.
    """
    config_hash = hash_config(config)

    if experiment_name is None:
        if isinstance(config, DictConfig) and "experiment_name" in config:
            experiment_name = config.experiment_name
        else:
            experiment_name = "experiment"

    dir_name = f"{experiment_name}_{config_hash}"
    output_dir = os.path.join(base_dir, dir_name)

    if create:
        new_dir = not os.path.exists(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        if new_dir:
            logger.info(f"Created output directory: {output_dir}")

        # Write config.yaml whenever missing (not only on first creation) so a
        # dir created by a job that died before this point still gets a config
        # when a later job with the same hash runs. Hash equality means content
        # is identical either way.
        config_path = os.path.join(output_dir, "config.yaml")
        if not os.path.exists(config_path):
            if isinstance(config, DictConfig):
                with open(config_path, "w") as f:
                    f.write(OmegaConf.to_yaml(config))
            else:
                with open(config_path, "w") as f:
                    yaml.dump(config, f, default_flow_style=False)

    return output_dir


def find_matching_output_dir(
    config: Union[dict[str, Any], DictConfig],
    base_dir: str = "./outputs",
) -> str | None:
    """Find an existing output directory whose hash matches *config*.

    Returns the path, or ``None`` if no match is found.
    """
    config_hash = hash_config(config)

    if not os.path.exists(base_dir):
        return None

    for dir_name in os.listdir(base_dir):
        if dir_name.endswith(config_hash):
            return os.path.join(base_dir, dir_name)

    return None