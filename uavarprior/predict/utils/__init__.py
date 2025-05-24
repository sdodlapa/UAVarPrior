"""
Utility functions for prediction module.
"""
import numpy as np

def get_reverse_complement_encoding(encoding, bases_arr=None, complementary_dict=None):
    """
    Get the reverse complement encoding of a sequence encoding.
    
    Parameters
    ----------
    encoding : numpy.ndarray
        One-hot encoded sequence.
    bases_arr : numpy.ndarray, optional
        Array of bases, typically ["A", "C", "G", "T"].
    complementary_dict : dict, optional
        Dictionary mapping bases to their complements.
        
    Returns
    -------
    numpy.ndarray
        Reverse complemented encoding.
    """
    if encoding is None:
        return None
        
    if len(encoding.shape) == 1:
        # Handle 1D sequence
        return np.flip(encoding)
    else:
        # Handle 2D one-hot encoding
        return np.flip(encoding, axis=0)

def _truncate_sequence(sequence, length):
    """
    Truncate a sequence to the specified length.
    
    Parameters
    ----------
    sequence : str
        Sequence to truncate.
    length : int
        Length to truncate to.
        
    Returns
    -------
    str
        Truncated sequence.
    """
    if len(sequence) <= length:
        return sequence
    else:
        # Cut from the middle
        start = (len(sequence) - length) // 2
        end = start + length
        return sequence[start:end]
