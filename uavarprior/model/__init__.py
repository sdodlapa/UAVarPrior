'''
Collection of models and wrappers
'''

from .nn import NonStrandSpecific
from .nn import DanQ
from .nn import DeeperDeepSEA
from .nn import HeartENN
from .nn import DeepSEA
from .nn import MultiNetWrapper

from .wrappers import UniSeqMWrapper

from .utils import loadModel
from .utils import loadModelFromFile

__all__ = ['NonStrandSpecific', 
           "DanQ",
           "DeeperDeepSEA",
           "DeepSEA",
           "HeartENN",
           # "deepcpg",
           'UniSeqMWrapper', 
           'MultiNetWrapper', 
           'loadModel',
           'loadModelFromFile']

import importlib
def loadNnModule(className):
    '''
    Load network module by class name
    '''
    if className == 'DanQ':
        return importlib.import_module('uavarprior.model.nn.danQ')
    elif className == 'DeeperDeepSEA':
        return importlib.import_module('uavarprior.model.nn.deeper_deepsea')
    elif className == 'DeepSEA':
        return importlib.import_module('uavarprior.model.nn.deepsea')
    elif className == 'Sei':
        return importlib.import_module('uavarprior.model.nn.sei')
    elif className == 'SeiHalf':
        return importlib.import_module('uavarprior.model.nn.sei_half')
    elif className == 'HeartENN':
        return importlib.import_module('uavarprior.model.nn.heatenn')
    elif className == 'SeqCnnL2h128' or className == 'CnnL3h128':
        return importlib.import_module('uavarprior.model.nn.deepcpg_dna')
    else:
        raise ValueError("Unrecognized network class {0}".format(className))
    
    
def loadWrapperModule(className):
    '''
    Load model wrapper module by class name
    '''
    if className == 'UniSeqMWrapper':
        return importlib.import_module('uavarprior.model.wrappers.uni_seq')
    else:
        raise ValueError("Unrecognized model wrapper class {0}".format(className))