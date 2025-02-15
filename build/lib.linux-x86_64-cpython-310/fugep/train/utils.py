'''
Created on May 23, 2021

@author: jsun
'''
from collections import OrderedDict


class LossTracker:
    '''
    Track loss when batches used in training and validation
    '''
    def __init__(self):
        self._loss = 0  # accumulated loss
        self._nItems = 0 # number of samples the produced that loss in _loss
    
    def add(self, loss, nItems):
        ''' 
        add the loss produced by a batch of nItems samples
        '''
        self._loss += loss
        self._nItems += nItems
    
    def getAveLoss(self):
        '''
        Return the average loss
        '''
        return self._loss / self._nItems
    
    def reset(self):
        '''
        Reset the track
        '''
        self._loss = 0
        self._nItems = 0


def load_model_from_state_dict(state_dict, model):
    """
    Loads model weights that were saved to a file previously by `torch.save`.
    This is a helper function to reconcile state dict keys where a model was
    saved with/without torch.nn.DataParallel and now must be loaded
    without/with torch.nn.DataParallel.

    Parameters
    ----------
    state_dict : collections.OrderedDict
        The state of the model.
    model : torch.nn.Module
        The PyTorch model, a module composed of submodules.

    Returns
    -------
    torch.nn.Module \
        The model with weights loaded from the state dict.

    Raises
    ------
    ValueError
        If model state dict keys do not match the keys in `state_dict`.

    """
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']

    model_keys = model.state_dict().keys()
    state_dict_keys = state_dict.keys()

    if len(model_keys) != len(state_dict_keys):
        try:
            model.load_state_dict(state_dict, strict=False)
            return model
        except Exception as e:
            raise ValueError("Loaded state dict does not match the model "
                "architecture specified - please check that you are "
                "using the correct architecture file and parameters.\n\n"
                "{0}".format(e))

    new_state_dict = OrderedDict()
    for (k1, k2) in zip(model_keys, state_dict_keys):
        value = state_dict[k2]
        try:
            new_state_dict[k1] = value
        except Exception as e:
            raise ValueError(
                "Failed to load weight from module {0} in model weights "
                "into model architecture module {1}. (If module name has "
                "an additional prefix `model.` it is because the model is "
                "wrapped in `selene_sdk.utils.NonStrandSpecific`. This "
                "error was raised because the underlying module does "
                "not match that expected by the loaded model:\n"
                "{2}".format(k2, k1, e))
    model.load_state_dict(new_state_dict)
    return model