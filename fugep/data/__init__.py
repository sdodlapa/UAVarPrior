
'''
Implementation of data handling
'''

from .utils import formatChrom

from .targets import Target
from .targets import GenomicFeatures
from .targets import DNAMethylTargets

from .sequences import Sequence
from .sequences import sequence_to_encoding
from .sequences import encoding_to_sequence
from .sequences import get_reverse_encoding
from .sequences import Genome
from .sequences import Proteome

__all__ = ['formatChrom',
           "Target", 
           "GenomicFeatures", 
           'DNAMethylTargets',
           "Sequence", 
           "Genome", 
           "Proteome", 
           "sequence_to_encoding",
           "encoding_to_sequence", 
           "get_reverse_encoding"]