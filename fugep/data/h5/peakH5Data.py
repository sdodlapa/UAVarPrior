'''
Prepare data in h5 format to train models for predicting peak-type events, 
e.g., chromatin accessibility, TF binding, and histone modification

Created on Apr 30, 2021

@author: Javon
'''

import os
import sys

import argparse
import logging
import numpy as np
from multiprocessing import Pool
import timeit

from ...utils import initialize_logger
from ...utils import load_features_list
from ...utils import make_dir
from ..sequences import Genome
from ..targets import GenomicFeatures
from .utils import DataStat
from .utils import seqWndRadius
from .utils import DataChunk

class App(object):

    def run(self, args):
        self.name = os.path.basename(args[0])
        parser = self.create_parser(self.name)
        self.opts = parser.parse_args(args[1:])
        return self.main()

    def create_parser(self, name):
        p = argparse.ArgumentParser(
            prog = name,
            formatter_class = argparse.ArgumentDefaultsHelpFormatter,
            description = 'Creates training and testing data for peak-type events.')

        # I/O
        p.add_argument(
            '--intvs',
            help = 'Bed file with genome intervals that are used to create'
            ' training and testing examples. It is assumed the intervals'
            ' are sorted according to chromosome and coordinates',
            required = True)
        p.add_argument(
            '--peaks',
            help = 'The bgziped sorted peak bed file. It is assumed this file has been'
            ' indexed with tabix and the index file is in the same folder.',
            required = True)
        p.add_argument(
            '--features',
            help='The file providing the list of features for which there are peaks'
            ' in the bed file given by argument peaks.',
            required = True)
        p.add_argument(
            '--refSeq',
            help='Fasta file providing the reference sequence',
            required = True)
        p.add_argument(
            '--blackListRegion',
            help='Can be hg19 or hg38 or a path to a bed file that contains '
            'blacklist region')
        p.add_argument(
            '-o', '--outDir',
            help='Output directory',
            required = True)

        g = p.add_argument_group('advanced arguments')
        g.add_argument(
            '--featureThreshold',
            type=float,
            help='Minimum overlap with a peak to label an interval positive')
        g.add_argument(
            '--includeNegative',
            type=bool,
            default=False,
            help='Indicate whether to include examples are negative for all features')
        g.add_argument(
            '--seqLen',
            type=int,
            default=2000,
            help='Length of the sequence used as input to the network')
        g.add_argument(
            '--binLen',
            type=int,
            default=200,
            help='Length of the bin centered in the input sequence. The overlap'
            ' between this bin and peak is used to label a sample.')
        g.add_argument(
            '--chunkSize',
            type=int,
            default=65536,
            help='Maximum number of samples per output file except the last one')
        g.add_argument(
            '--minChunkSize',
            type=int,
            default=32768,
            help='Minimum number of samples in the last output file. If not enough '
            'to make a separate output file, the remaining will be added to '
            'the chunk before.')
        g.add_argument(
            '--compressSequence',
            type=bool,
            default=False,
            help='Indicate whether to compress sequence')
        g.add_argument(
            '--compressTarget',
            type=bool,
            default=False,
            help='Indicate whether to compress target (i.e., label)')
#         g.add_argument(
#             '--seed',
#             help='Seed of random number generator',
#             type=int,
#             default=0)
        g.add_argument(
            '--verbose',
            default=False,
            help='More detailed log messages')
        g.add_argument(
            '--logFile',
            help='Write log messages to file')
        return p

    def retrieve(self, intv):
        initialize_logger(self.logFile, name=self.name, verbosity=self.logVerb)
        logger = logging.getLogger(self.name)
        refSeq = Genome(self.opts.refSeq, self.opts.blackListRegion)
        intvFileHdl = open(intv, 'r')
        for index, line in enumerate(intvFileHdl):
            cols = line.strip().split('\t')
            chrom = cols[0]
            start = int(cols[1])
            end = int(cols[2])
            centPos = start + int((end - start) / 2)

            # compute the center bin
            binStart = centPos - self.binStartRadius
            binEnd = centPos + self.binEndRadius
            # label the center bin
            time = timeit.default_timer()
            binLbl = self.target.getFeatureData(
                chrom, binStart, binEnd)
            logger.debug('Time to extract feature:%s' % (timeit.default_timer() - time))
            if not self.opts.includeNegative and np.sum(binLbl) == 0:
                logger.info("No features found in region surrounding "
                                 "region \"{0}\" position {1}. Skip.".format(
                    chrom, centPos))
                continue

            wndStart = centPos - self.wndStartRadius
            windowEnd = centPos + self.wndEndRadius
            time = timeit.default_timer()
            seq = refSeq.get_encoding_from_coords(chrom, wndStart, windowEnd)
            logger.debug('Time to extract sequence:%s' % (timeit.default_timer() - time))
            if seq.shape[0] == 0:
                logger.info("Full sequence centered at region \"{0}\" position "
                                 "{1} could not be retrieved. Skip.".format(
                    chrom, centPos))
                continue
            elif np.sum(seq == 1) / float(seq.shape[0]) < 0.70:
                logger.info("Over 30% of the bases in the sequence centered "
                                 "at region \"{0}\" position {1} are ambiguous ('N'). "
                                 "Skip.".format(chrom, centPos))
                continue

            # a valid sample, save it
            self.nSamps += 1
            if self.curChunk is None:
                # initialize a chunk
                self.curChunk = DataChunk(self.nSamps - 1, self.opts.seqLen,
                                     self.features, self.opts.chunkSize)

            # add to the current chunk
            self.curChunk.add(seq, binLbl, chrom, np.array([binStart, binEnd]))
            # update the statistics
            time = timeit.default_timer()
            self.datastat.add(binLbl, chrom)
            logger.debug('Time to add statistic:%s' % (timeit.default_timer() - time))

            if (self.curChunk.isFull()):
                # save the self.preChunk
                if self.preChunk is not None:
                    self.preChunk.saveToH5(self.opts.outDir, compressSeq=self.opts.compressSequence,
                                      compressLbl=self.opts.compressTarget)
                    self.nChunks += 1
                    logger.info('{0} chunks in {1} have been saved ...'.format(self.nChunks, chrom))

                self.preChunk = self.curChunk
                self.curChunk = None

        # save chunk and stat
        if self.preChunk is not None:
            if self.curChunk is not None and \
                    self.curChunk.getSize() < self.opts.minChunkSize:
                # merge with previous chunk
                self.preChunk.merge(self.curChunk)
                self.curChunk = None

            self.preChunk.saveToH5(self.opts.outDir, compressSeq=self.opts.compressSequence,
                              compressLbl=self.opts.compressTarget)
            self.nChunks += 1
            logger.info('{0} chunks in {1} have been saved ...'.format(self.nChunks, chrom))

        if self.curChunk is not None:
            self.curChunk.saveToH5(self.opts.outDir, compressSeq=self.opts.compressSequence,
                              compressLbl=self.opts.compressTarget)
            self.nChunks += 1
            logger.info('{0} chunks in {1} have been saved ...'.format(self.nChunks, chrom))

        return self.datastat

    def main(self):
        # validate the input
        if (self.opts.seqLen + self.opts.binLen) % 2 != 0:
            raise ValueError(
                "Sequence length of {0} with a center bin length of {1} "
                "is invalid. These 2 inputs should both be odd or both be "
                "even.".format(self.opts.seqLen, self.opts.binLen))

#         if self.opts.seed is not None:
#             np.random.seed(self.opts.seed)
#
        make_dir(self.opts.outDir)
        if self.opts.verbose:
            self.logVerb = 2
        else:
            self.logVerb = 1
        if self.opts.logFile is None:
            self.logFile = os.path.join(self.opts.outDir, 'run.log')
        else:
            self.logFile = self.opts.logFile
        initialize_logger(self.logFile, name = self.name, verbosity = self.logVerb)
        self.logger = logging.getLogger(self.name)
        self.logger.debug(self.opts)

        # load reference genome
        self.logger.info('Loading reference sequence ...')
        refSeq = Genome(self.opts.refSeq, self.opts.blackListRegion)
        # load features
        self.logger.info('Loading features ...')
        self.features = load_features_list(self.opts.features)
        # load targets (peaks)
        self.logger.info('Loading peaks ...')
        self.target = GenomicFeatures(self.opts.peaks, self.features,
            feature_thresholds = self.opts.featureThreshold)

        # initialization for computing the data statistics
        self.datastat = DataStat(self.features, refSeq.get_chrs())

        # compute bin radius
        self.binStartRadius, self.binEndRadius = seqWndRadius(self.opts.binLen)
        # compute window radius
        self.wndStartRadius, self.wndEndRadius = seqWndRadius(self.opts.seqLen)

        self.logger.info('labeling intervals ...')
        self.preChunk, self.curChunk = None, None
        self.nSamps, self.nChunks = 0, 0

        intvs = [self.opts.intvs + '/' + intv for intv in os.listdir(self.opts.intvs)]
        # retrieve data and statistics in parallel
        with Pool(processes=len(os.listdir(self.opts.intvs))) as pool:
            stats = pool.map(self.retrieve, intvs)

        self.datastat.merge(stats)

        self.datastat.saveToCSV(os.path.join(self.opts.outDir, 'meta.csv'))

        self.logger.info('Done!')

        return 0


if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
