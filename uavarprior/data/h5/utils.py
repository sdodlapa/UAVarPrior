'''
Created on May 1, 2021

@author: jsun
'''

import pandas as pd
import numpy as np
import os
import h5py as h5
from collections import OrderedDict

from ..utils import sortChrom

class DataStat():
    '''
    A class to hold statistics of the data for training and testing
    '''
    STAT_ITEMS = ['pos', 'neg']

    def __init__(self, features, chroms):
        self._features = features

        self._itemIdx = OrderedDict()
        for idx, val in enumerate(self.STAT_ITEMS):
            self._itemIdx[val] = idx
        self._chromIdx = OrderedDict()
        for idx, val in enumerate(sortChrom(chroms)):
            self._chromIdx[val] = idx

        self._stat = np.zeros((len(features), len(self.STAT_ITEMS), len(chroms)),
                              dtype=np.int32)

    def add(self, lbls, chrom):
        '''
        Add an example to the statistics
        '''
        chromIdx = self._chromIdx[chrom]
        for item in self.STAT_ITEMS:
            itemIdx = self._itemIdx[item]
            if item == 'pos':
                self._stat[:, itemIdx, chromIdx] += (lbls == 1)
            elif item == 'neg':
                self._stat[:, itemIdx, chromIdx] += (lbls == 0)

    def merge(self, stats):
        '''
        Merge statistics from all chromosomes
        '''
        for stat in stats:
            self._stat = np.add(self._stat, stat._stat)

    def saveToCSV(self, filepath):
        '''
        Save data statistics to a file
        '''
        statDf = {}
        for iFeat in range(len(self._features)):
            feat = self._features[iFeat]
            statDf[feat] = pd.DataFrame(self._stat[iFeat],
                                        index=self._itemIdx.keys(), columns=self._chromIdx.keys())

        combStat = None
        for feat, fStat in statDf.items():
            # stat per chr
            colTtl = fStat.sum(0)
            colPosRate = fStat.loc['pos'] / colTtl
            chrStat = fStat.append(colTtl, ignore_index=True)
            chrStat = chrStat.append(colPosRate, ignore_index=True)

            # stat of overall data
            rowTtl = fStat.sum(1)
            ttl = rowTtl.sum()
            posRate = rowTtl[0] / ttl
            statOvl = rowTtl.append(pd.Series([ttl, posRate]),
                                    ignore_index=True)

            # stat column
            statItem = pd.Series(fStat.index.to_list() + ['ttl', 'pos_rate'])
            # feature column
            featCol = pd.Series([feat] * statItem.size)

            # combine all data
            fStatEx = pd.concat([featCol, statItem, chrStat, statOvl],
                                axis=1, ignore_index=True)
            fStatEx.columns = ['feature', 'stat'] + fStat.columns.to_list() \
                              + ['overall']

            if combStat is None:
                combStat = fStatEx
            else:
                combStat = pd.concat([combStat, fStatEx], ignore_index=True)
        # save to file
        combStat.to_csv(filepath, index=False)

    # calculate sequence radius


def seqWndRadius(seqLen):
    wndRadius = int(seqLen / 2)
    wndStartRadius = wndRadius
    wndEndRadius = wndRadius
    if seqLen % 2 != 0:
        wndEndRadius += 1
    return wndStartRadius, wndEndRadius


class DataChunk():
    """
    Chunk of data for peak-type events, including data of sequence,
    target, chrom, coor and feature

    Data chunks for methylation and iteraction data can be created by
    inheriting from this class
    """

    def __init__(self, startIdx, seqLen, features, maxCapacity,
                 nCoor=2, nSeqBase=4):
        '''
        Initiate a new DataChunk object

        Parameters
        ----------------
        nCoor : number of elements in coor
        '''

        self._startIdx = startIdx
        self._seqLen = seqLen
        self._nSeqBase = nSeqBase
        self._features = features
        self._maxCapacity = maxCapacity

        # initialize data fields
        self._size = 0
        self._chrom = []
        if nCoor == 1:
            self._coor = np.zeros(maxCapacity, dtype=np.uint64)
        else:
            self._coor = np.zeros((maxCapacity, nCoor), dtype=np.uint64)
        self._seq = np.zeros((maxCapacity, seqLen, self._nSeqBase),
                             dtype=np.float16)
        self._lbl = np.zeros((maxCapacity, len(features)),
                             dtype=np.float16)

    def _expand(self, newCap):
        '''
        increase max capacity
        '''
        if self._maxCapacity >= newCap:
            return

        self._maxCapacity = newCap

        if len(self._coor.shape) == 1:
            newCoor = np.zeros(self._maxCapacity, dtype=np.uint64)
        else:
            newCoor = np.zeros((self._maxCapacity, self._coor.shape[1]), dtype=np.uint64)
        newCoor[:self._size] = self._coor
        self._coor = newCoor

        newSeq = np.zeros((self._maxCapacity, self._seqLen, self._nSeqBase),
                          dtype=np.float16)
        newSeq[:self._size] = self._seq
        self._seq = newSeq

        newLbl = np.zeros((self._maxCapacity, len(self._features)),
                          dtype=np.float16)
        newLbl[:self._size] = self._lbl
        self._lbl = newLbl

    def _merge(self, chunkToMerge):
        '''
        Merge another chunk assuming there is adequate space to hold
        the data in the given chunk to merge

        return
        ------------------
        int :
            the new size after merging
        '''
        newSize = self._size + chunkToMerge._size
        self._chrom = self._chrom + chunkToMerge._chrom
        self._coor[self._size:newSize] = chunkToMerge._coor[:chunkToMerge._size]
        self._seq[self._size:newSize] = chunkToMerge._seq[:chunkToMerge._size]
        self._lbl[self._size:newSize] = chunkToMerge._lbl[:chunkToMerge._size]

        return newSize

    def add(self, seq, lbl, chrom, coor):
        """
        Add an example to the chunk
        """
        if self._size >= self._maxCapacity:
            raise ValueError("Reach max data chunk capacity!")

        self._chrom.append(chrom)
        self._coor[self._size] = coor
        self._seq[self._size] = seq
        self._lbl[self._size] = lbl

        self._size += 1

    def isFull(self):
        '''
        Check if the chunk reaches its full capacity
        '''
        return self._size >= self._maxCapacity

    def getChr(self):
        '''
        Get the chromosome from which the data in the chunk come
        '''
        return self._chrom[0]

    def getSize(self):
        '''
        Get the size of the chunk, i.e., number of samples in the chunk
        '''
        return self._size

    def merge(self, chunkToMerge):
        """
        Merge the data in another chunk
        """
        if chunkToMerge._size == 0:
            return

        if self._maxCapacity < self._size + chunkToMerge._size:
            # expand
            self._expand(self._size + chunkToMerge._size)

        # merge
        self._size = self._merge(chunkToMerge)


    def saveToH5(self, outDir, targetDtype='uint8', keepFileOpen=False,
                 compressSeq=False, compressLbl=False):
        '''
        Save data in chunk to h5 file

        Parameters
        ----------
        targetDtype: whether to save targets(labels) as int or float. Default is 'Uint8'
        keepFileOpen: if True, h5 file will not be closed and
        the file handle will be returned. If false, the file
        will be closed and nothing will be returned

        compressSeq: whether to compress sequence

        compressLbl: whether to compress label(target)
        '''
        if self._size == 0:
            return

        if self._chrom[0].startswith('chr'):
            filename = '%s_%06d-%06d.h5' % (self._chrom[0], self._startIdx,
                                            self._startIdx + self._size - 1)
        else:
            filename = 'chr%s_%06d-%06d.h5' % (self._chrom[0], self._startIdx,
                                               self._startIdx + self._size - 1)
        filename = os.path.join(outDir, filename)
        h5File = h5.File(filename, 'w')

        # Write chromosome
        h5File.create_dataset('chrom', (self._size,), dtype='S5',
                              data=[n.encode("ascii", "ignore") for n in self._chrom], compression='gzip')

        # write coordinate
        h5File.create_dataset('coor', dtype='uint64',
                              data=self._coor[:self._size], compression='gzip')

        # write sequence
        if compressSeq:
            h5File.create_dataset('sequence', dtype='uint8',
                                  data=np.packbits(self._seq[:self._size] > 0, axis=1), compression='gzip')
        else:
            h5File.create_dataset('sequence', dtype='uint8', data=self._seq[:self._size], compression='gzip')

        # write sequence length
        h5File.create_dataset('sequence_length', dtype='uint16',
                              data=self._seqLen)

        # write label
        if compressLbl:
            h5File.create_dataset('targets', dtype=targetDtype,
                                  data=np.packbits(self._lbl[:self._size] > 0, axis=1), compression='gzip')
        else:
            h5File.create_dataset('targets', dtype=targetDtype, data=self._lbl[:self._size], compression='gzip')

        # write target length
        #         h5File.create_dataset('targets_length', dtype = 'uint32',
        #                   data = len(self._features))

        # write features
        h5File.create_dataset('features',
                              data=[n.encode("ascii", "ignore") for n in self._features], compression='gzip')

        if keepFileOpen:
            return h5File
        else:
            h5File.close()



