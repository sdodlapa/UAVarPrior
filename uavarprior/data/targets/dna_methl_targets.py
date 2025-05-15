'''
This module contains methods to query a set of DNA methylation profiles, 
which can be in the format of tsv or bedgraph

Created on May 13, 2021

@author: jsun
'''

import numpy as np
import os
from collections import namedtuple

from functools import wraps
from .target import Target
from ..utils import formatChrom
from torch.nn import init


MethylData = namedtuple('MethylData', ['pos', 'value'])

class DNAMethylTargets(Target):
    """
    Stores the dataset specifying position of cytocines and their 
    methlylation state.
    Accepts a tabix-indexed `*.bed` file with the following columns,
    in order:
    ::
        [chrom, start, end, strand, feature]


    Note that `chrom` is interchangeable with any sort of region (e.g.
    a protein in a FAA file). Further, `start` is 0-based. Lastly, any
    addition columns following the five shown above will be ignored.

    Parameters
    ----------
    inputPath : str
        Path to the directory holding all .spv (single position value) files.
    metaFile : str
        The name of the tab delimited file in inputPath that provides meta info of the 
        datasets. Two columns are expected, first column providing the ID of 
        the dataset, the second column specifying the name of the data file

    initUnpicklable : bool, optional
        Default is False. Delays initialization until a relevant method
        is called. This enables the object to be pickled after instantiation.
        `initUnpicklable` must be `False` when multi-processing is needed e.g.
        DataLoader. 
    
    binary : bool, optional, default is True
        Indicate whether to convert methylation value to binary (if not already
        in the input file)

    Attributes
    ----------
    data : tabix.open
        The data stored in a tabix-indexed `*.bed` file.
    n_features : int
        The number of distinct features.
    feature_index_dict : dict
        A dictionary mapping feature names (`str`) to indices (`int`),
        where the index is the position of the feature in `features`.
    index_feature_dict : dict
        A dictionary mapping indices (`int`) to feature names (`str`),
        where the index is the position of the feature in the input
        features.
    feature_thresholds : dict or None

        * `dict` - A dictionary mapping feature names (`str`) to thresholds\
        (`float`), where the threshold is the minimum overlap that a\
        feature annotation must have with a query region to be\
        considered a positive example of that feature.
        * `None` - No threshold specifications. Assumes that all features\
        returned by a tabix query are annotated to the query region.

    """
    
    METHYL_UNK = -1 # numeric code for unknown methylation status
    
    def __init__(self, inputPath, metaFile, initUnpicklable = False, 
                 binary = False):
        """
        Constructs a new `GenomicFeatures` object.
        """
        self._inputPath = inputPath
        self._metaFile = metaFile
        self._binary = binary

        self._initialized = False

        if initUnpicklable:
            self._unpicklableInit()

    def _unpicklableInit(self):
        if self._initialized:
            return
        
        self._methylData = dict()
        self._features = []
        metaFileHdl = open(os.path.join(self._inputPath, self._metaFile), 'r')
        for _, line in enumerate(metaFileHdl):
            cols = line.strip().split('\t')
            feat = cols[0].strip()
            self._features.append(feat)
            fileName = cols[1].strip()
            
            self._methylData[feat] = dict()
            curChrom, positions, values = None, None, None
            profFileHdl = open(os.path.join(self._inputPath, fileName), 'r')
            for _, dataLine in enumerate(profFileHdl):
                dataCols = dataLine.strip().split('\t')
                chrom = formatChrom(dataCols[0].strip())
                if curChrom is None or curChrom != chrom:
                    # a new chromosome
                    if positions is not None:
                        # save data from curChrom
                        if self._binary:
                            # convert to binary
                            valArr = np.ones(len(values), dtype = np.int8)
                            valArr[np.array(values) < 0.5] = 0
                        else:
                            valArr = np.array(values, dtype = np.float16)
                        self._methylData[feat][curChrom] = \
                            MethylData(np.array(positions, dtype = np.uint64),
                                       valArr)
                        
                    # initiate a new list
                    curChrom = chrom
                    positions, values = [], []
                    
                positions.append(int(dataCols[1].strip())) 
                values.append(float(dataCols[2].strip())) 
            
            # save the ending chromosome
            if positions is not None and len(positions) > 0:
                # save data from curChrom
                if self._binary:
                    # convert to binary
                    valArr = np.ones(len(values), dtype = np.int8)
                    valArr[np.array(values) < 0.5] = 0
                else:
                    valArr = np.array(values, dtype = np.float16)
                self._methylData[feat][curChrom] = \
                    MethylData(np.array(positions, dtype = np.uint64),
                               valArr)
                
                
        self._initialized = True

    def init(func):
        # delay initialization to allow multiprocessing
        @wraps(func)
        def dfunc(self, *args, **kwargs):
            self._unpicklableInit()
            return func(self, *args, **kwargs)
        return dfunc


    @init
    def hasData(self, chrom, pos):
        """
        Determines whether the CpG at the given position, for which
        there is data available in any of the methylation profile.

        Parameters
        ----------
        chrom : str
            The name of the region (e.g. '1', '2', ..., 'X', 'Y').
        pos : int
            The 0-based genome position of the query CpG.
            
        Returns
        -------
        bool
            `True` if this is,
            `False` otherwise.
        """
        methylData = self.getFeatureData(chrom, pos)
        if np.sum(methylData) == self.METHYL_UNK * len(self._features):
            return True
        else:
            return False
    
    @init
    def getAllPositions(self):
        '''
        Get all positions of Cytosines for which there are data available 
        in at least one profile
        
        Returns
        -------
        dict(str, []):
            dict(chrom, positions)
        '''
        allPos = dict()
        # pool all unique positions from all profiles
        for feat in self._features:
            for chrom, mData in self._methylData[feat].items():
                if chrom in allPos:
                    allPos[chrom] = allPos[chrom].union(mData.pos.tolist())
                else:
                    allPos[chrom] = set(mData.pos.tolist())
            
        # sort by position coordinate within each chromosome
        for chrom in allPos:
            allPos[chrom] = sorted(allPos[chrom])
        
        return allPos
    
    @init
    def getChromPositions(self, chrom):
        '''
        Similar to getAllPositions - gets all the positions of target chromosome for which there are data
        available in at least one profile
        
       Returns
        -------
        dict(str, []):
            dict(chrom, positions)
        
        '''
        chrPos = dict()
        for feat in self._features:
            if chrom in self._methylData[feat].keys():
                if chrom in chrPos:

                    chrPos[chrom] = chrPos[chrom].union(
                        self._methylData[feat][chrom].pos.tolist())
                else:
                    chrPos[chrom] = set(
                        self._methylData[feat][chrom].pos.tolist())

        return chrPos

    @init
    def getChroms(self):
        '''
        Gets all chormosomes present in the data
        
        Returns
        -------
        list
            list of chromosomes - ['1', '2', ....., 'X', 'Y']
            
        '''
        chroms = set()
        for feat in self._features:
            chroms = chroms.union(self._methylData[feat].keys())
        chroms = list(chroms)
        return chroms
    
        
    @init
    def getFeatures(self):
        '''
        Get all features (i.e., the ID of the methylation profiles)
        '''
        return self._features
    
    @init
    def getFeatureData(self, chrom, pos):
        """
        Computes which features overlap with the given region.

        Parameters
        ----------
        chrom : str
            The name of the region (e.g. '1', '2', ..., 'X', 'Y').
        pos : int
            The 0-based genome position of the query cytosine.

        Returns
        -------
        numpy.ndarray
            A target vector of size `self.n_features` where the `i`th
            position is equal to one if the `i`th feature is positive,
            and zero otherwise.

            NOTE: If we catch a `tabix.TabixError`, we assume the error was
            the result of there being no features present in the queried region
            and return a `numpy.ndarray` of zeros.

        """
                
        dataOut = np.ones(len(self._features)) * self.METHYL_UNK
        for iFeat in range(len(self._features)):
            feat = self._features[iFeat]
            featProf = self._methylData[feat]
            if chrom in featProf:
                idx = np.searchsorted(featProf[chrom].pos, pos)
                if idx < len(featProf[chrom].pos) and \
                    featProf[chrom].pos[idx] == pos:
                    # there is data available
                    dataOut[iFeat] = featProf[chrom].value[idx]
        return dataOut
        
    @init
    def getMethylInWnd(self, chrom, center, ext):    
        '''
        Retrieve methylation data of cytosine in the windown centered at 'center'
        and extended to both sides by 'ext' cytosines
        The distance of surrounding cytosine to the centered one is also returned
        
        Parameters
        ----------
        chrom : str
            The name of the region (e.g. '1', '2', ..., 'X', 'Y').
        center : int
            The 0-based genome position of the query cytosine.
        ext: int
            The number of cyosines from both sides of the centered, for which 
            methyl data to be retrieved and distant to be calculated
    
        Returns
        -------
        numpy.ndarray, methylation data
            N X M, where N is the number of features, M is 2*ext
        numpy.ndarray, distance
            N X M, where N is the number of features, M is 2*ext
        '''    
        methyl = np.ones((len(self._features), 2 * ext)) * self.METHYL_UNK
        dist = np.ones((len(self._features), 2 * ext))  * self.METHYL_UNK
        
        for iFeat in range(len(self._features)):
            feat = self._features[iFeat]
            featProf = self._methylData[feat]
            if chrom in featProf:
                idx = np.searchsorted(featProf[chrom].pos, center)
                
                # left side
                leftEndIdx = idx
                leftStartIdx = max(0, leftEndIdx - ext)
                leftLen = leftEndIdx - leftStartIdx
                if leftLen > 0:
                    methyl[iFeat][(ext - leftLen):ext] = \
                        featProf[chrom].value[leftStartIdx:leftEndIdx]
                    dist[iFeat][(ext - leftLen):ext] = \
                        center - featProf[chrom].pos[leftStartIdx:leftEndIdx]
                
                # right side
                if idx == len(featProf[chrom].pos):
                    # very right side, i.e., no positions on the right side of center
                    continue
                rightStartIdx = leftEndIdx
                if featProf[chrom].pos[idx] == center:
                    # skip the center
                    rightStartIdx += 1
                rightEndIdx = min(len(featProf[chrom].pos), rightStartIdx + ext)
                rightLen = rightEndIdx - rightStartIdx
                if rightLen > 0:
                    methyl[iFeat][ext:(ext + rightLen)] = \
                        featProf[chrom].value[rightStartIdx:rightEndIdx]
                    dist[iFeat][ext:(ext + rightLen)] = \
                        featProf[chrom].pos[rightStartIdx:rightEndIdx] - center
            
        return methyl, dist
        