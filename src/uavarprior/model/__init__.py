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
from typing import Dict, Any

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


def get_model(config: Dict[str, Any]):
    """Factory function for model instantiation (moved from uavarprior/models)."""
    model_name = config.get("name")
    if not model_name:
        raise ValueError("Model name not specified in configuration")
    try:
        module = loadNnModule(model_name)
    except Exception as e:
        raise ValueError(f"Model module not found: {model_name}") from e
    try:
        model_class = getattr(module, "Model")
    except AttributeError:
        # Fallback to class named after the model
        if hasattr(module, model_name):
            model_class = getattr(module, model_name)
        else:
            raise ValueError(f"Model class not found in module: {model_name}")
    # Only pass the intended constructor args, not the config keys
    class_args = config.get("classArgs", {}) or {}
    # Handle legacy key naming for Sei
    if model_name in ("Sei", "SeiHalf") and "n_targets" in class_args:
        class_args["n_genomic_features"] = class_args.pop("n_targets")
    model_instance = model_class(**class_args)
    # Provide a toUseCuda method expected by StandardSGDTrainer
    import types
    def _toUseCuda(self):
        return self.cuda()
    model_instance.toUseCuda = types.MethodType(_toUseCuda, model_instance)
    return model_instance