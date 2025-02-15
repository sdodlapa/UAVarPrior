'''
Implementations for performing mutation sensitivity analysis for
DNA squence via in-silico mutagenesis with applying models trained
for predicting various functional genomic events
'''

from .peak import PeakISMSeqAnalyzer

__all__ = ['PeakISMSeqAnalyzer']