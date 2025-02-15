'''
Implementation of loss functions

Created on May 23, 2021

@author: jsun
'''

import torch
# from torch.tensor import Tensor # no module torch.tensor in latest version (7/28/2021)
from torch import Tensor
import torch.nn as nn


def weightedBCELoss(prediction: Tensor, target: Tensor, weight: Tensor = None):
    '''
    weighted binary cross entropy loss. Reduction is mean.
    Samples with 0 weight are ignored during the reduction 
    
    Return:
    Tensor : average loss over the batch
    Tensor : sum of the loss of the batch
    Tensor : number of effective terms in the loss, i.e., number of non-zero weights
    '''
    temp_target = target.clone()
    temp_target[target < 0] = 0
    temp_pred = prediction.clone()
    temp_pred[target < 0] = 0
    loss = nn.functional.binary_cross_entropy(temp_pred, temp_target,
                      weight = weight, reduction = 'none')

    # loss = nn.functional.binary_cross_entropy(prediction, target,
    #                  weight = weight, reduction = 'none')
    sumOfLoss = torch.sum(loss)
    if weight is not None:
        nEffTerms = torch.count_nonzero(weight)
    else:
        nEffTerms = torch.tensor(torch.numel(target))
    if nEffTerms == 0:
        aveOfLoss = torch.tensor(0)
    else:
        aveOfLoss = torch.div(sumOfLoss, nEffTerms)
    batchLoss = torch.div(sumOfLoss, target.shape[0])
    return aveOfLoss, batchLoss, sumOfLoss, nEffTerms
    
    