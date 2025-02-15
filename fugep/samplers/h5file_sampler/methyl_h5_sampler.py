'''
Created on Apr 30, 2021

@author: jsun
'''

import numpy as np
import torch

from .h5_sampler import H5Sampler
from ..dataloader.h5 import H5DataLoader
from ..utils import getSWeight

class _MethylCollator(object):
    '''
    collate retrieved data to create a mini-batch
    '''
    def __init__(self, seqLen, featsInH5, cWeights = None, iFeatsPred = None,
                 binarize_labels = None, nSeqBase = 4, valueOfMissing = None):
        self._seqLen = seqLen
        self._nSeqBase = nSeqBase
        self._featsInH5 = featsInH5
        self._binarize_labels = binarize_labels
        self._cWeights = cWeights
        self._iFeatsPred = iFeatsPred
        self._valueOfMissing = valueOfMissing
    
    def __call__(self, batch):
        ### Arrays to hold sequences and targets
        sequences = np.zeros((len(batch), self._seqLen, self._nSeqBase), dtype = np.float32)
        targets = np.zeros((len(batch), len(self._featsInH5)))

        # retrieve data
        for iSamp in range(len(batch)):
            samp = batch[iSamp]
            sequences[iSamp] = np.eye(self._nSeqBase)[samp['inputs']]
            targets[iSamp] = samp['outputs']
        
        # keep only target data of features to predict
        if self._iFeatsPred is not None:
            targets = targets[:, self._iFeatsPred]
        if self._binarize_labels:
            targets[(targets >= 0) & (targets <= 0.5)] = 0
            targets[targets > 0.5] = 1
            
        # compute the sample weights
        if self._iFeatsPred is not None:
            features = [self._featsInH5[i] for i in self._iFeatsPred]
        else:
            features = self._featsInH5
        weights = getSWeight(targets, features = features, 
                            cWeights = self._cWeights,
                            valueOfMissing = self._valueOfMissing)

        return (torch.from_numpy(sequences),
                torch.from_numpy(targets),
                torch.from_numpy(weights))


class MethylH5Sampler(H5Sampler):
    '''
    Class for sampling from H5 files of DNA (CpG) methylation data
    '''
    # names of datasets to load from H5 file
    _NAME_OF_SEQ = ['inputs']
    _NAME_OF_TRT = ['outputs']
    _NAME_OF_CPG = ['cpg/methyl',  'cpg/dist'] 

    def __init__(self,
                 h5FileDir,
                 train = None,
                 validate = None,
                 batchSize = 64,
                 features = None,  # if none, all features in h5 will be included
                 test = None,
                 mode = "train",
                 weightSampByCls = True,
                 valOfMisInTarget = None,
                 binarize_labels = None,
                 seed = None,
                 nWorkers = 1,
                 save_datasets = [],
                 output_dir = None):
        """
        Constructs a new `IntervalH5Sampler` object.
        """
        super(MethylH5Sampler, self).__init__(
            h5FileDir = h5FileDir,
            train = train,
            validate = validate,
            features = features,
            test = test,
            mode = mode,
            weightSampByCls = weightSampByCls,
            valOfMisInTarget = valOfMisInTarget,
            binarize_labels = binarize_labels,
            save_datasets = save_datasets,
            output_dir = output_dir)

        dataToLoad = self._NAME_OF_SEQ + self._NAME_OF_TRT
            
        self._dataloaders['train'] = \
            H5DataLoader(self._train,
                 dataToLoad, self._featsInH5,
                 collateFunc = _MethylCollator(self._seqLen, self._featsInH5,
                                            cWeights = self._cWeights,
                                            iFeatsPred = self._iFeatsPred,
                                            binarize_labels = binarize_labels,
                                            nSeqBase = self._N_SEQ_BASE,
                                            valueOfMissing = valOfMisInTarget),
                 seed = seed,
                 nWorkers = nWorkers,
                 batchSize = batchSize,
                 shuffle = True,
            )
        self._iterators['train'] = iter(self._dataloaders['train'])
        
        self._dataloaders['validate'] = \
            H5DataLoader(self._validate,
                 dataToLoad, self._featsInH5,
                 collateFunc=_MethylCollator(self._seqLen, self._featsInH5,
                                             cWeights=self._cWeights,
                                             iFeatsPred=self._iFeatsPred,
                                             binarize_labels=binarize_labels,
                                             nSeqBase=self._N_SEQ_BASE,
                                             valueOfMissing=valOfMisInTarget),
                 seed = seed,
                 nWorkers = nWorkers,
                 batchSize = batchSize,
                 shuffle = False,
            )
        self._iterators['validate'] = iter(self._dataloaders['validate'])
        
        if test is not None:
            self._dataloaders['test'] = \
            H5DataLoader(self._test,
                 dataToLoad, self._featsInH5,
                 collateFunc=_MethylCollator(self._seqLen, self._featsInH5,
                                             cWeights=self._cWeights,
                                             iFeatsPred=self._iFeatsPred,
                                             binarize_labels=binarize_labels,
                                             nSeqBase=self._N_SEQ_BASE,
                                             valueOfMissing=valOfMisInTarget),
                 seed = seed,
                 nWorkers = nWorkers,
                 batchSize = batchSize,
                 shuffle = False,
            )
            self._iterators['test'] = iter(self._dataloaders['test'])

    def sample(self, batchSize = 64, mode = None):
        """
        Fetches a mini-batch of the data from the sampler.

        Parameters
        ----------
        batchSize : int, optional
            Default is 1. The size of the batch to retrieve.
        mode : str, optional
            Default is None. The operating mode that the object should run in.
            If None, will use the current mode `self.mode`.
        """
        mode = mode if mode else self.mode
        self._setBatchSize(batchSize, mode = mode)
        try:
            seq, targets, weights = next(self._iterators[mode])
        except StopIteration:
            #If DataLoader iterator reaches its length, reinitialize for training
            if mode in ['train', 'validate']:
                self._iterators[mode] = iter(self._dataloaders[mode])
                seq, targets, weights = next(self._iterators[mode])
            else:
                return None
        
        dataOut = {'sequence': seq.numpy(),
                'targets': targets.numpy(),
                'weights': weights.numpy()}
        return dataOut    

    def getDataAndTargets(self, batchSize, nSamps = None, mode = None):
        """
        This method fetches a subset of the data from the sampler,
        divided into batches. This method also allows the user to
        specify what operating mode to run the sampler in when fetching
        the data.

        Parameters
        ----------
        batchSize : int
            The size of the batches to divide the data into.
        nSamps : int or None, optional
            Default is None. The total number of samples to retrieve.
            If `nSamps` is None, if a FileSampler is specified for the 
            mode, the number of samplers returned is defined by the FileSample, 
            or if a Dataloader is specified, will set `nSamps` to 32000 
            if the mode is `validate`, or 640000 if the mode is `test`. 
            If the mode is `train` you must have specified a value for 
            `nSamps`.
        mode : str, optional
            Default is None. The operating mode that the object should run in.
            If None, will use the current mode `self.mode`.
        """
        
        mode = mode if mode is not None else self.mode
        self._setBatchSize(batchSize, mode = mode)
        
        dataAndTargets = []
        trgtsMat = []
        
        if nSamps is None:
            # retrieve all samples managed by the corresponding sampler
            # reset the iterator
            self._iterators[mode] = iter(self._dataloaders[mode])
            while True:
                try:
                    seq, targets, weights = next(self._iterators[mode])
                    dataBatch = {'sequence': seq.numpy(),
                                'targets': targets.numpy(),
                                'weights': weights.numpy()}
                    dataAndTargets.append(dataBatch)
                    trgtsMat.append(dataBatch['targets'])
                except StopIteration:
                    break
        else:
            count = batchSize
            while count < nSamps:
                dataBatch = self.sample(batchSize = batchSize, mode = mode)
                dataAndTargets.append(dataBatch)
                trgtsMat.append(dataBatch['targets'])
                count += batchSize
            remainder = batchSize - (count - nSamps)
            dataBatch = self.sample(batchSize = remainder, mode=mode)
            dataAndTargets.append(dataBatch)
            trgtsMat.append(dataBatch['targets'])
        
        trgtsMat = np.vstack(trgtsMat)
        return dataAndTargets, trgtsMat
            
    def getValidationSet(self, batchSize, nSamps = None):
        """
        This method returns a subset of validation data from the
        sampler, divided into batches.

        Parameters
        ----------
        batchSize : int
            The size of the batches to divide the data into.
        nSamps : int, optional
            Default is None. The total number of validation examples to
            retrieve. If `nSamps` is None,
            then if a FileSampler is specified for the 'validate' mode, the
            number of samplers returned is defined by the FileSample,
            or if a Dataloader is specified, will set `nSamps` to
            32000.

        Returns
        -------
        sequences_and_targets, targets_matrix : \
        tuple(list(dict()), numpy.ndarray)
            Tuple containing the list of sequence-target-weight dicts, as well
            as a single matrix with all targets in the same order.
            Note that `sequences_and_targets` sequence elements are of
            the shape :math:`B \\times L \\times N` and its target
            elements are of the shape :math:`B \\times F`, where
            :math:`B` is `batchSize`, :math:`L` is the sequence length,
            :math:`N` is the size of the sequence type's alphabet, and
            :math:`F` is the number of features. Further,
            `target_matrix` is of the shape :math:`S \\times F`, where
            :math:`S = nSamps`.

        Raises
        ------
        ValueError
            If no test partition of the data was specified during
            sampler initialization.
        """
        return self.getDataAndTargets(
            batchSize, nSamps, mode = "validate")

    def getTestSet(self, batchSize, nSamps = None):
        """
        This method returns a subset of testing data from the
        sampler, divided into batches.

        Parameters
        ----------
        batchSize : int
            The size of the batches to divide the data into.
        nSamps : int or None, optional
            Default is None. The total number of test examples to
            retrieve. If `nSamps` is None,
            then if a FileSampler is specified for the 'test' mode, the
            number of samplers returned is defined by the FileSample,
            or if a Dataloader is specified, will set `nSamps` to
            640000.

        Returns
        -------
        sequences_and_targets, targets_matrix : \
        tuple(list(dict()), numpy.ndarray)
            Tuple containing the list of sequence-target-weight dicts, as well
            as a single matrix with all targets in the same order.
            Note that `sequences_and_targets` sequence elements are of
            the shape :math:`B \\times L \\times N` and its target
            elements are of the shape :math:`B \\times F`, where
            :math:`B` is `batchSize`, :math:`L` is the sequence length,
            :math:`N` is the size of the sequence type's alphabet, and
            :math:`F` is the number of features. Further,
            `target_matrix` is of the shape :math:`S \\times F`, where
            :math:`S = nSamps`.

        Raises
        ------
        ValueError
            If no test partition of the data was specified during
            sampler initialization.
        """
        return self.getDataAndTargets(
            batchSize, nSamps, mode = "test")


        