"""
Data module for UAVarPrior.

This module provides data handling utilities.
"""

# Import the Genome class from FuGEP for compatibility
try:
    from fugep.data import Genome
except ImportError:
    try:
        from uavarprior.data.genome import Genome
    except ImportError:
        import logging
        logging.warning("Could not import Genome class. Some functionality may be unavailable.")
        
        # Create a stub class for compilation
        class Genome:
            def __init__(self, input_path=None):
                self.input_path = input_path
                raise NotImplementedError("Genome class not available")