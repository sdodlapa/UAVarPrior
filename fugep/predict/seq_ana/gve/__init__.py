'''
Implementations for genetic variant effect evaluation
with applying models trained
for predicting various functional genomic events
'''

from .peak import PeakGVarEvaluator
from .methyl import MethylVarEvaluator

__all__ = ['PeakGVarEvaluator', 'MethylVarEvaluator']