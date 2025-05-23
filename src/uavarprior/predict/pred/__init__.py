'''
Implementation for predicing functional genomic events
'''

from .peak import PeakEventPredictor
from .evn_pred import EventPredictor
from .methyl import MethylEventPredictor

__all__ = ['PeakEventPredictor',
           'MethylEventPredictor']