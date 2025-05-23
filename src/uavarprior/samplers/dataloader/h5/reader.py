'''
This module provides the base class of H5Reader

Created on Apr 30, 2021

@author: jsun
'''

import h5py as h5
import numpy as np
import six
from six.moves import range

class H5Reader():
    '''
    Base class of H5Reader
    '''
    def __init__(self, h5Files, nameOfData, features, seed = None, shuffle = True):
        '''
        Parameters
        -------------------------
        h5Files: the list of h5 files from which data are retrieved
        nameOfData: the name of the dataset to retrieve form the h5 files
        '''
        
        self._h5Files = h5Files
        self._nameOfData = nameOfData
        self._shuffle = shuffle
        self._seed = seed
        self.features = features
        
        if self._seed is not None:
            np.random.seed(self._seed)
    
    def __iter__(self):
        # permute the ordering of the file
        if self._shuffle:
            np.random.shuffle(self._h5Files)
            
        # iterate through files
        for h5File in self._h5Files:
            h5FileHdl = h5.File(h5File, 'r')
            
            datasets = dict()
            for name in self._nameOfData:
                if name == 'inputs':
                    datasets[name] = h5FileHdl[name]['dna'][:]
                if name == 'outputs':
                    targets = []
                    for feat in self.features:
                        targets.append(h5FileHdl[name][feat][:])
                    datasets[name] = np.transpose(np.array(targets)) ## nSamples X nFeatures

            nSampInFile = datasets['inputs'].shape[0]
            
            if self._shuffle:
                # Shuffle data within the entire file, which requires reading
                # the entire file into memory
                idx = np.arange(nSampInFile)
                np.random.shuffle(idx)
                for name, value in six.iteritems(datasets):
                    datasets[name] = value[:len(idx)][idx]
            
            # iterate through samples in the file
            for iSamp in range(nSampInFile):
                dataOfSamp = dict()
                for name in self._nameOfData:
                    dataOfSamp[name] = datasets[name][iSamp]
                yield  dataOfSamp    
            
            h5FileHdl.close()
            

                
                
                
                