# This file turns this directory into the `uavarprior.utils` package.
# It allows imports such as `from uavarprior.utils.logging import setup_logging`.

from .logging import setup_logging
import logging

def initialize_logger(output_path, name='uavarprior', verbosity=2):
    """
    Initialize UAVarPrior logger using the package logging setup.
    """
    level = logging.DEBUG if verbosity >= 2 else (logging.INFO if verbosity == 1 else logging.WARN)
    # Configure console and file handlers
    setup_logging(level=level, log_file=output_path)

__all__ = ['setup_logging', 'initialize_logger']
