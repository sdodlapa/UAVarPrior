'''
Prepare data in h5 format to train models for predicting DNA methylation

Created on Apr 30, 2021

@author: Javon
'''

import os
import sys

import argparse
import logging
import time
import warnings
import numpy as np
import pandas as pd
from multiprocessing import Pool
from fugep.utils import initialize_logger
from fugep.utils import make_dir
from fugep.data.sequences import Genome
from fugep.data.targets import DNAMethylTargets
from fugep.data.h5.utils import DataStat
from fugep.data.h5.utils import seqWndRadius
from fugep.data.h5.utils import DataChunk
from fugep.data.utils import isBedgraph
from fugep.data.utils import isBinary
from fugep.data.utils import formatChrom
from fugep.data.utils import sampleByChrom


def readCpgProf(filename, chroms=None, nSamp=None,
                toRound=False, sort=True, nSampPerChrom=None):
    """Read CpG profile from TSV or bedGraph file.

    Reads CpG profile from either tab delimited file with columns
    `chrom`, `pos`, `value` (i.e., .spv), or bedGraph file. `value` columns contains
    methylation states, which can be binary or continuous.

    Parameters
    ----------
    filenamne: str
        Path of file.
    chroms: list
        List of formatted chromosomes to be read, e.g. ['1', 'X'].
    nSamp: int
        Maximum number of sample in total.
    toRound: bool
        If `True`, round methylation states in column 'value' to zero or one.
    sort: bool
        If `True`, sort by rows by chromosome and position.
    nSampPerChrom: int
        Maximum number of sample per chromosome.

    Returns
    -------
    :class:`pandas.DataFrame`
         :class:`pandas.DataFrame` with columns `chrom`, `pos`, `value`.

    Note
    -------
    Modified from read_cpg_profile in DeepCpG
    """

    # load data
    if isBedgraph(filename):
        usecols = [0, 1, 3]
        skiprows = 1
    else:
        usecols = [0, 1, 2]
        skiprows = 0
    dtype = {usecols[0]: np.str, usecols[1]: np.int32, usecols[2]: np.float16}
    nrows = None
    if chroms is None and nSampPerChrom is None:
        nrows = nSamp
    d = pd.read_table(filename, header=None, comment='#', nrows=nrows,
                      usecols=usecols, dtype=dtype, skiprows=skiprows)
    d.columns = ['chrom', 'pos', 'value']

    # validate the data
    if np.any((d['value'] < 0) | (d['value'] > 1)):
        raise ValueError('Methylation values must be between 0 and 1!')

    # use data from given chromosomes
    d['chrom'] = formatChrom(d['chrom'])
    if chroms is not None:
        if not isinstance(chroms, list):
            chroms = [str(chroms)]
        d = d.loc[d.chrom.isin(chroms)]
        if len(d) == 0:
            raise ValueError('No data available for selected chromosomes!')

    if nSampPerChrom is not None:
        d = sampleByChrom(d, nSampPerChrom)

    if nSamp is not None and len(d) > nSamp:
        d = d.iloc[:nSamp]

    if sort:
        d.sort_values(['chrom', 'pos'], inplace=True)

    if toRound:
        d['value'] = np.round(d.value)

    if isBinary(d['value']):
        d['value'] = d['value'].astype(np.int8)

    return d


def posDictReader(posDict):
    '''
    Load positions from a dictionary like the one returned by
    getAllPositions function of DNAMethylTargets
    '''
    for chrom in posDict:
        for pos in posDict[chrom]:
            yield chrom, pos


def posFileReader(posFile):
    '''
    Load genome positions from a tab deliminated file, first column
    gives the chromosome, second column gives coordinate
    '''
    fileHdl = open(posFile, 'r')
    for _, line in enumerate(fileHdl):
        cols = line.strip().split('\t')
        chrom = formatChrom(cols[0])
        pos = int(cols[1])

        yield chrom, pos


class MethylDataChunk(DataChunk):
    """
    Chunk of data for DNA methylation event
    """

    def __init__(self, startIdx, seqLen, features, maxCapacity,
                 wndLen=25, nSeqBase=4):
        super(MethylDataChunk, self).__init__(
            startIdx=startIdx,
            seqLen=seqLen,
            features=features,
            maxCapacity=maxCapacity,
            nCoor=1,
            nSeqBase=nSeqBase)

        self._wndLen = wndLen
        if self._wndLen == 0:
            # no methylation data of surrounding CpGs is needed
            return

        self._wndMethyl = np.zeros((maxCapacity, len(features), wndLen),
                                   dtype=np.uint8)
        self._wndDist = np.zeros((maxCapacity, len(features), wndLen),
                                 dtype=np.uint32)

    def _expand(self, newCap):
        '''
        increase max capacity
        '''
        if self._maxCapacity >= newCap:
            return

        super(MethylDataChunk, self)._expand(newCap)
        if self._wndLen == 0:
            return

        # deal with data elements for surrounding CpGs
        newWndMethyl = np.zeros((self._maxCapacity, len(self._features),
                                 self._wndLen), dtype=np.uint8)
        newWndMethyl[:self._size] = self._wndMethyl
        self._wndMethyl = newWndMethyl

        newWndDist = np.zeros((self._maxCapacity, len(self._features),
                               self._wndLen), dtype=np.uint32)
        newWndDist[:self._size] = self._wndDist
        self._wndDist = newWndDist

    def _merge(self, chunkToMerge):
        '''
        Merge another chunk assuming there is adequate space to hold
        the data in the given chunk to merge

        return
        ------------------
        int :
            the new size after merging
        '''
        newSize = super(MethylDataChunk, self)._merge(chunkToMerge)
        if self._wndLen == 0:
            return newSize

        self._wndMethyl[self._size:newSize] = \
            chunkToMerge._wndMethyl[:chunkToMerge._size]
        self._wndDist[self._size:newSize] = \
            chunkToMerge._wndDist[:chunkToMerge._size]

        return newSize

    def add(self, seq, lbl, chrom, coor,
            wndMethyl=None, wndDist=None):
        """
        Add an example to the chunk
        """
        super(MethylDataChunk, self).add(
            seq=seq, lbl=lbl, chrom=chrom, coor=coor)

        if self._wndLen == 0:
            return

        if wndMethyl is None or wndDist is None:
            raise ValueError('Inputs: wndMethyl and wndDist can not be none')

        self._wndMethyl[self._size - 1] = wndMethyl
        self._wndDist[self._size - 1] = wndDist

    def saveToH5(self, outDir, targetDtype, keepFileOpen=False,
                 compressSeq=False):
        '''
        Save the data in the chunk to a H5 file
        '''
        h5File = super(MethylDataChunk, self).saveToH5(
            outDir=outDir,
            targetDtype=targetDtype,
            keepFileOpen=True,
            compressSeq=compressSeq,
            compressLbl=False)

        if self._wndLen > 0:
            cpgGrp = h5File.create_group('cpg')
            cpgGrp.create_dataset('methyl', dtype=targetDtype,
                                  data=self._wndMethyl[:self._size])
            cpgGrp.create_dataset('dist', dtype='uint32',
                                  data=self._wndDist[:self._size])

        if keepFileOpen:
            return h5File
        else:
            h5File.close()


class App(object):

    def run(self, args):
        self.name = os.path.basename(args[0])
        parser = self.create_parser(self.name)
        self.opts = parser.parse_args(args[1:])
        return self.main()

    def create_parser(self, name):
        p = argparse.ArgumentParser(
            prog=name,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description='Creates training and testing data for peak-type events.')

        # I/O
        p.add_argument(
            '--cpgProfDir',
            help='Directory that contains methylation profiles in spv format')
        p.add_argument(
            '--coorBase',
            help='Base of position coordinate, either 0 or 1. Optional, 1 by default',
            type=int,
            default=0)
        p.add_argument(
            '--cpgProfMeta',
            help='Name of the meta file in the cpgProfDir that provides info '
                 'of the profiles. Two tab separated columns Two columns are expected, '
                 'first column providing the ID of the profile, the second column '
                 'specifying the name of the data file')
        p.add_argument(
            '--cpgPos',
            help='File specifying positions of CpG sites that are to be predicted.'
                 ' Optional, if missing, only CpG sites that are observed in at least one of'
                 ' the given profiles will be used. It is assumed rows are sorted by chromosome'
                 ' and position. Two tab separated columns are anticipated, first one specifying the'
                 ' chromosome, second one giving the coordinate.')
        p.add_argument(
            '--cpgWLen',
            help='If provided, extract `cpgWLen`//2 neighboring CpG sites',
            type=int,
            default=0)
        p.add_argument(
            '--cpgCov',
            help='Minimum CpG coverage. Only use CpG sites for which the true'
                 ' methylation state is known in at least that many profiles.',
            type=int,
            default=1)

        p.add_argument(
            '--refSeq',
            help='Fasta file providing the reference sequence',
            required=True)
        p.add_argument(
            '--blackListRegion',
            help='Can be hg19 or hg38 or a path to a bed file that contains '
                 'blacklist region')
        p.add_argument(
            '-o', '--outDir',
            help='Output directory',
            required=True)

        g = p.add_argument_group('advanced arguments')
        g.add_argument(
            '--chromos',
            nargs='+',
            help='Chromosomes that are used')
        g.add_argument(
            '--seqLen',
            type=int,
            default=2001,
            help='Length of the sequence used as input to the network')
        g.add_argument(
            '--targetDtype',
            type=str,
            default='uint8',
            help='Data type of targets to be saved')
        g.add_argument(
            '--isTargetBinary',
            type=bool,
            default=False,
            help='Whether target labels binary')
        g.add_argument(
            '--chunkSize',
            type=int,
            default=32768,
            help='Maximum number of samples per output file except the last one')
        g.add_argument(
            '--minChunkSize',
            type=int,
            default=16384,
            help='Minimum number of samples in the last output file. If not enough '
                 'to make a separate output file, the remaining will be added to '
                 'the chunk before.')
        g.add_argument(
            '--compressSequence',
            type=bool,
            default=True,
            help='Indicate whether to compress sequence')
        g.add_argument(
            '--verbose',
            help='More detailed log messages',
            action='store_true')
        g.add_argument(
            '--logFile',
            help='Write log messages to file')
        return p

    def retrieve(self, chrom):
        print(f'Chromosome: {chrom}')
        initialize_logger(self.logFile, name=self.name, verbosity=self.logVerb)
        logger = logging.getLogger(self.name)
        refSeq = Genome(self.opts.refSeq, self.opts.blackListRegion)

        # compute sequence window radius
        seqWndSRadius, seqWndERadius = seqWndRadius(self.opts.seqLen)

        if self.opts.cpgPos is not None:
            # TODO: update posGen code for cpgPos argument as well
            posGen = posFileReader(self.opts.cpgPos)
        else:
            print(f'Getting posGen')
            posGen = posDictReader(self.target.getChromPositions(chrom))

        logger.info("Data extraction for {0} is started...".format(chrom))
        preChunk, curChunk = None, None
        nSamps, nChunks = 0, 0
        for chrom, pos in posGen:
            # get the label
            posLbl = self.target.getFeatureData(chrom, pos)
            if np.sum(posLbl != -1) < self.opts.cpgCov:
                logger.debug("Not enough profiles with data at {0} on chromosome \"{1}\", "
                             "{2} available, {3} expected. Skip.".format(
                    pos, chrom, np.sum(posLbl != -1), self.opts.cpgCov))
                continue

            adjPos = pos - self.opts.coorBase
            wndStart = adjPos - seqWndSRadius
            wndEnd = adjPos + seqWndERadius
            seq = refSeq.get_encoding_from_coords(chrom, wndStart, wndEnd)
            if seq.shape[0] == 0:
                logger.info("Full sequence centered at chromosome {0} position "
                            "{1} could not be retrieved. Skip.".format(chrom, pos))
                continue

            elif np.sum(seq == 1) / float(seq.shape[0]) < 0.70:
                logger.info("Over 30% of the bases in the sequence centered "
                            "at region \"{0}\" position {1} are ambiguous ('N'). "
                            "Skip.".format(chrom, pos))
                continue
            elif Genome.encoding_to_sequence(seq[seqWndSRadius:(seqWndSRadius + 2)]) != 'CG' and \
                    Genome.encoding_to_sequence(seq[(seqWndSRadius - 1):(seqWndSRadius + 1)]) != 'CG':
                logger.info('No CpG site at chromosome {0} position {1}. Skip.'.format(chrom, pos))
                # warnings.warn('No CpG site at chromosome  %s at position %d. Skip ...' % (chrom, nChunks))
                continue

            # Time taken for seq, ambiguous(N), and CG position check
            # t1 = time.perf_counter()
            # seq = refSeq.get_encoding_from_coords(chrom, wndStart, wndEnd)
            # t2 = time.perf_counter()
            # logger.info("Time taken for seq extraction {0}: ". format(t2-t1))
            #
            # t1 = time.perf_counter()
            # np.sum(seq == 1) / float(seq.shape[0]) < 0.70
            # t2 = time.perf_counter()
            # logger.info("Time taken to check ambiguous position (N) {0}: ".format(t2 - t1))
            #
            # t1 = time.perf_counter()
            # Genome.encoding_to_sequence(seq[seqWndSRadius:(seqWndSRadius + 2)]) != 'CG' and \
            # Genome.encoding_to_sequence(seq[(seqWndSRadius - 1):(seqWndSRadius + 1)]) != 'CG'
            # t2 = time.perf_counter()
            # logger.info("Time taken for CG check {0}: ".format(t2 - t1))


            # a valid sample, save it
            if (curChunk is not None and curChunk.getChr() != chrom) or \
                    (preChunk is not None and preChunk.getChr() != chrom):
                # start a new chromosome, save both open chunks
                if curChunk is not None and \
                        curChunk.getSize() < self.opts.minChunkSize and \
                        preChunk is not None:
                    # merge with previous chunk
                    preChunk.merge(curChunk)
                    curChunk = None

                if preChunk is not None:
                    # if self.opts.targetDtype == 'np.float':
                    #     self.opts.targetDtype = np.float32
                    preChunk.saveToH5(self.opts.outDir, self.opts.targetDtype, compressSeq=self.opts.compressSequence)
                    nChunks += 1
                    logger.info('{0} {1} chunks have been saved ...'.format(nChunks, chrom))
                    warnings.warn('%d chunks of chrom %s have been saved ...' % (nChunks, chrom))
                    print(f'{nChunks} {chrom} chunks have been saved ...')

                if curChunk is not None:
                    # if self.opts.targetDtype == 'np.float':
                    #     self.opts.targetDtype = np.float32
                    curChunk.saveToH5(self.opts.outDir, self.opts.targetDtype, compressSeq=self.opts.compressSequence)
                    nChunks += 1
                    logger.info('{0} ({1}) chunks have been saved ...'.format(nChunks, chrom))
                    warnings.warn('%d chunks of chrom %s have been saved ...' % (nChunks, chrom))

                preChunk, curChunk, nSamps = None, None, 0

            nSamps += 1
            if curChunk is None:
                # initialize a chunk
                curChunk = MethylDataChunk(nSamps - 1, self.opts.seqLen,
                                           self.target.getFeatures(), self.opts.chunkSize, wndLen=self.opts.cpgWLen)

            wndMethyl, wndDist = None, None
            if self.opts.cpgWLen > 0:
                cpgWExt = self.opts.cpgWLen // 2
                wndMethyl, wndDist = self.target.getMethylInWnd(chrom, pos, cpgWExt)
            # add to the current chunk
            curChunk.add(seq, posLbl, chrom, pos, wndMethyl, wndDist)

            # update the statistics
            self.dataStat.add(posLbl, chrom)

            if (curChunk.isFull()):
                # save the preChunk
                if preChunk is not None:
                    # if self.opts.targetDtype == 'np.float':
                    #     self.opts.targetDtype = np.float132
                    preChunk.saveToH5(self.opts.outDir, self.opts.targetDtype, compressSeq=self.opts.compressSequence)
                    nChunks += 1
                    logger.info('{0} ({1}) chunks have been saved ...'.format(nChunks, chrom))

                preChunk = curChunk
                curChunk = None

        # save ending chunk and stat
        if preChunk is not None:
            if curChunk is not None and \
                    curChunk.getSize() < self.opts.minChunkSize:
                # merge with previous chunk
                preChunk.merge(curChunk)
                curChunk = None

            # if self.opts.targetDtype == 'np.float':
            #     self.opts.targetDtype = np.float32
            preChunk.saveToH5(self.opts.outDir, self.opts.targetDtype, compressSeq=self.opts.compressSequence)
            nChunks += 1
            logger.info('{0} ({1}) chunks have been saved ...'.format(nChunks, chrom))

        if curChunk is not None:
            # if self.opts.targetDtype == 'np.float':
            #     self.opts.targetDtype = np.float32
            curChunk.saveToH5(self.opts.outDir, self.opts.targetDtype, compressSeq=self.opts.compressSequence)
            nChunks += 1
            logger.info('{0} ({1}) chunks have been saved ...'.format(nChunks, chrom))
        return self.dataStat

    def main(self):
        # validate the input
        if self.opts.seqLen and self.opts.seqLen % 2 == 0:
            raise '--seqLen must be odd!'
        if self.opts.cpgWLen and self.opts.cpgWLen % 2 != 0:
            raise '--cpgWLen must be even!'

        make_dir(self.opts.outDir)
        if self.opts.verbose:
            self.logVerb = 2
        else:
            self.logVerb = 1
        if self.opts.logFile is None:
            self.logFile = os.path.join(self.opts.outDir, 'run.log')
        else:
            self.logFile = self.opts.logFile
        initialize_logger(self.logFile, name=self.name, verbosity=self.logVerb)
        self.logger = logging.getLogger(self.name)
        self.logger.debug(self.opts)

        # load reference genome
        self.logger.info('Loading reference sequence ...')
        refSeq = Genome(self.opts.refSeq, self.opts.blackListRegion)
        # load targets (methylation profiles)
        self.logger.info('Loading methylation profiles ...')
        self.target = DNAMethylTargets(self.opts.cpgProfDir, self.opts.cpgProfMeta,
                                  initUnpicklable=True, binary=False)

        # chromosomes for which data to be extracted
        if self.opts.chromos:
            chromInc = self.opts.chromos
        else:
            chromInc = [formatChrom(chrom) for chrom in refSeq.get_chrs()]
            # chromInc = [chrom for chrom in chromInc if (chrom in ['X', 'Y'])
            #             or (chrom.isnumeric())] # include known chromosomes only (numerically coded, and X and Y)

            chromInc = [chrom for chrom in chromInc if chrom.isnumeric()] # only numerically coded chroms
            print(f'Chromosomes used to extract: {chromInc}')

        # initialization for computing the data statistics
        self.dataStat = DataStat(self.target.getFeatures(), chromInc)

        self.logger.info('labeling CpGs ...')

        # retrieve data and statistics in parallel
        with Pool(processes=os.cpu_count()) as pool:
            stats = pool.map(self.retrieve, chromInc)

        self.dataStat.merge(stats)
        self.dataStat.saveToCSV(os.path.join(self.opts.outDir, 'meta.csv'))

        self.logger.info('Done!')
        return 0


if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
