# This file turns this directory into the `uavarprior.utils` package.
# It allows imports such as `from uavarprior.utils.logging import setup_logging`.

from .logging import setup_logging
import logging
import numpy as np
import os
import sys

def initialize_logger(output_path, name='uavarprior', verbosity=2):
    """
    Initialize UAVarPrior logger using the package logging setup.
    """
    level = logging.DEBUG if verbosity >= 2 else (logging.INFO if verbosity == 1 else logging.WARN)
    # Configure console and file handlers
    setup_logging(level=level, log_file=output_path)

def get_indices_and_probabilities(interval_lengths, indices):
    """
    Given interval lengths and indices, compute sampling weights.
    """
    select_lens = np.array(interval_lengths)[indices]
    total = float(np.sum(select_lens))
    weights = select_lens / total
    keep = [i for i, w in zip(indices, weights) if w > 1e-10]
    if len(keep) == len(indices):
        return indices, weights.tolist()
    return get_indices_and_probabilities(interval_lengths, keep)

def load_features_list(input_path):
    """Read feature names from a file, one per line."""
    with open(input_path) as f:
        return [line.strip() for line in f]

def make_dir(dirname):
    """Create directory if it does not exist."""
    if os.path.exists(dirname):
        return False
    os.makedirs(dirname)
    return True

__all__ = ['setup_logging', 'initialize_logger', 'get_indices_and_probabilities', 'load_features_list', 'make_dir']
