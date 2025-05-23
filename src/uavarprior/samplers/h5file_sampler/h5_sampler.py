'''
Created on May 5, 2021

@author: jsun
'''
"""
This module provides the `MultiSampler` class, which accepts
either an online sampler or a file sampler for each mode of
sampling (train, test, validation).
MultiSampler is a subclass of Sampler.
"""
from abc import ABCMeta
import os
import glob  
import h5py as h5  

from ..sampler import Sampler
from ..utils import calcCWeightByH5File

class H5Sampler(Sampler, metaclass = ABCMeta):
    """
    This sampler draws samples from individual file samplers or data loaders
    that corresponds to training, validation, and testing (optional) modes.
    MultiSampler calls on the correct file sampler or data loader to draw
    samples for a given mode. Example file samplers are under
    `fugep.samplers.file_samplers` and example data loaders are under
    `fugep.samplers.dataloaders`.

    MultiSampler can use either file samplers or data loaders for
    different modes. Using data loaders for some modes while using file samplers
    for other modes are also allowed. The file samplers parse data files
    (e.g. bed, mat, or hdf5). The data loaders provide multi-worker iterators
    that draw samples from online samplers (i.e. on-the-fly sampling). As data
    loaders support parallel sampling, they are generally recommended for
    sampling speed.

    Parameters
    ----------
    h5FileDir: the directory that holds all the input h5 files
    train : chromosomes used for training
    validate : chromosomes used for validation
    test : chromosomes used for testing
    mode : str, optional
        Default is "train". Must be one of `{train, validate, test}`. The
        starting mode in which to run the sampler.
    save_datasets : list(str) or None, optional
        Default is None. Currently, we are only including this parameter
        so that `MultiSampler` is consistent with the `Sampler` interface.
        The save dataset functionality for MultiSampler has not been
        defined yet.
    output_dir : str or None, optional
        Default is None. Only used if the sampler has any data or
        logging statements to save to file. Currently not used in
        `MultiSampler`.

    Attributes
    ----------
    modes : list(str)
        A list of the modes that the object may operate in.
    mode : str or None
        Default is `None`. The current mode that the object is operating in.

    """
    
    _N_SEQ_BASE = 4
    
    def __init__(self,
                 h5FileDir,
                 train = None,
                 validate = None,
                 features = None,  # if none, all features in h5 will be included
                 test = None,
                 mode="train",
                 weightSampByCls = True,
                 clsWeights = None,
                 valOfMisInTarget = None, # value representing missing in target
                 binarize_labels=None,
                 save_datasets = [],
                 output_dir = None):
        """
        Constructs a new `H5Sampler` object.
        """
        # identify all h5 files under the given h5FileDir
        h5Files = glob.glob(os.path.join(h5FileDir, '*.h5'))
        # obtain the list of files for training, validation and testing
        self._train = []
        self._validate = []
        self._test = []
        for h5File in h5Files:
            chrom = os.path.basename(h5File).split('_')[0]
            if train is not None and chrom in train:
                self._train.append(h5File)
            elif validate is not None and chrom in validate:
                self._validate.append(h5File)
            elif test is not None and chrom in test:
                self._test.append(h5File)
        
        if mode == 'train':
            if len(self._train) == 0:
                raise ValueError('No chromosomes are specified for training')
            if len(self._validate) == 0:
                raise ValueError('No chromosomes are specified for validation')
            sampH5File = self._train[0]
        elif mode == 'validate':
            if len(self._validate) == 0:
                raise ValueError('No chromosomes are specified for validation')
            sampH5File = self._validate[0]
        elif mode == 'test':
            if len(self._test) == 0:
                raise ValueError('No chromosomes are specified for testing')
            sampH5File = self._test[0]

        # retrieve features from a H5 file
        with h5.File(sampH5File, 'r') as fh:
            self._featsInH5 = [feat for feat in fh['outputs'].keys()]
            self._seqLen = fh['inputs']['dna'].shape[1]
        
        # initialize members in the super class    
        super(H5Sampler, self).__init__(
            self._featsInH5,
            save_datasets = save_datasets,
            output_dir = output_dir,
            valOfMisInTarget = valOfMisInTarget,
            binarize_labels = binarize_labels,
            clsWeights = clsWeights)
        
        # add test operation mode
        if self._test:
            self.modes.append('test')
        
        # set the features to predict for
        if features is not None: 
            self._features = features
            self._iFeatsPred = [i for i, feat in enumerate(self._featsInH5) if feat in features]
        else:
            self._features = self._featsInH5
            self._iFeatsPred = None
        
        self._indexToFeature = {i: f for (i, f) in enumerate(self._features)}
        
        self._dataloaders = {} # to be initialized in subclass
        self._iterators = {} # to be initialized in subclass

        self.mode = mode
        
        # set class weights
        if weightSampByCls and self._cWeights is None and len(self._train) > 0:
            self._cWeights = calcCWeightByH5File(self._train, features = features,
                                            binarize_labels = self._binarize_labels,
                                            valueOfMissing = self._valOfMisInTarget)

    def setMode(self, mode):
        """
        Sets the sampling mode.

        Parameters
        ----------
        mode : str
            The name of the mode to use. It must be one of
            `Sampler.BASE_MODES` ("train", "validate") or "test" if
            the test data is supplied.

        Raises
        ------
        ValueError
            If `mode` is not a valid mode.

        """
        if mode not in self.modes:
            raise ValueError(
                "Tried to set mode to be '{0}' but the only valid modes are "
                "{1}".format(mode, self.modes))
        self.mode = mode

    def _setBatchSize(self, batchSize, mode=None):
        """
        Sets the batch size for DataLoader for the specified mode,
        if the specified batchSize does not equal the current batchSize.
        Parameters
        ----------
        batchSize : int
            The batch size for the mode.
        mode : str, optional
            Default is None. The  mode to set batchSize
            If None, will use the current mode `self.mode`.
        """
        if mode is None:
            mode = self.mode

        if self._dataloaders[mode] is not None:
            batch_size_matched = True
            if self._dataloaders[mode].batch_sampler is not None:
                if self._dataloaders[mode].batch_sampler.batch_size != batchSize:
                    self._dataloaders[mode].batch_sampler.batch_size = batchSize
                    batch_size_matched = False
            else:
                if self._dataloaders[mode].batch_size != batchSize:
                    self._dataloaders[mode].batch_size = batchSize
                    batch_size_matched = False

            if not batch_size_matched:
                print("Reset data loader for mode {0} to use the new batch "
                      "size: {1}.".format(mode, batchSize))
                self._iterators[mode] = iter(self._dataloaders[mode])

    def getFeatureByIndex(self, index):
        return self._indexToFeature[index]
    
    def getSequenceLength(self):
        return self._seqLen

