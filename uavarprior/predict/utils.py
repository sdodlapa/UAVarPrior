"""
Prediction specific utility functions.
"""
import math
import numpy as np
import os

def createFilePathWithPrefix(prefix, fileName):
    '''
    Create a path to a file with the given prefix and fileName.
    when the prefix is a path to a directory, the the path to the file
    would be {prefix}/{fileName}, otherwise {prefix}-{fileName}
    '''
    
    dirPath = None
    namePrefix = None
    if not os.path.isdir(prefix):
        dirPath, namePrefix = os.path.split(prefix)
    else:
        dirPath = prefix
    if namePrefix is not None:
        fileName = "{0}-{1}".format(
            namePrefix, fileName)
    filePath = os.path.join(dirPath, fileName)
    return filePath

def get_reverse_complement(allele, complementary_base_dict):
    """
    Get the reverse complement of the input allele.

    Parameters
    ----------
    allele : str
        The sequence allele
    complementary_base_dict : dict(str)
        The dictionary that maps each base to its complement

    Returns
    -------
    str
        The reverse complement of the allele.

    """
    if allele == '*' or allele == '-' or len(allele) == 0:
        return '*'
    a_complement = []
    for a in allele:
        a_complement.append(complementary_base_dict[a])
    return ''.join(list(reversed(a_complement)))


def get_reverse_complement_encoding(allele_encoding,
                                    bases_arr,
                                    complementary_base_dict):
    """
    Get the reverse complement of the input allele one-hot encoding.

    Parameters
    ----------
    allele_encoding : numpy.ndarray
        The sequence allele encoding, :math:`L \\times 4`
    bases_arr : list(str)
        The base ordering for the one-hot encoding
    complementary_base_dict : dict(str: str)
        The dictionary that maps each base to its complement

    Returns
    -------
    np.ndarray
        The reverse complement encoding of the allele, shape
        :math:`L \\times 4`.

    """
    base_ixs = {b: i for (i, b) in enumerate(bases_arr)}
    complement_indices = [
        base_ixs[complementary_base_dict[b]] for b in bases_arr]
    return allele_encoding[:, complement_indices][::-1, :]


def _pad_sequence(sequence, to_length, unknown_base):
    diff = (to_length - len(sequence)) / 2
    pad_l = int(np.floor(diff))
    pad_r = math.ceil(diff)
    sequence = ((unknown_base * pad_l) + sequence + (unknown_base * pad_r))
    return str.upper(sequence)


def _truncate_sequence(sequence, to_length):
    start = int((len(sequence) - to_length) // 2)
    end = int(start + to_length)
    return str.upper(sequence[start:end])
