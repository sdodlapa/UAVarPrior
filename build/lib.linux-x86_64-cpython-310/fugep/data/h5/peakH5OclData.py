'''
Created on May 17, 2021
@author: madshiri
'''

import os
import sys
import glob
import shutil
import random
import logging
import argparse
import h5py as h5
import numpy as np
from fugep.utils import make_dir
from fugep.utils import initialize_logger
from fugep.data.h5.utils import DataChunk


class App(object):

    def run(self, args):
        name = os.path.basename(args[0])
        parser = self.create_parser(name)
        opts = parser.parse_args(args[1:])
        return self.main(name, opts)

    def create_parser(self, name):
        p = argparse.ArgumentParser(
            prog=name,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description='Sampling positive examples of h5 data for one-class classification.')

        # I/O
        p.add_argument(
            '--h5FilesDir',
            help='',
            required=True)
        p.add_argument(
            '--posRate',
            help='in percentage',
            default='10',
            required=True)
        p.add_argument(
            '--featToSamp',
            help='Path to a list of features in a text file',
            required=True)
        p.add_argument(
            '-o', '--outDir',
            help='Output directory',
            required=True)
        g = p.add_argument_group('advanced arguments')
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
        g.add_argument(
            '--verbose',
            default=False,
            help='More detailed log messages')
        g.add_argument(
            '--logFile',
            help='Write log messages to file')

        return p

    def main(self, name, opts):
        # create and specify output directorires
        outDir = opts.outDir
        if os.path.isdir(outDir) is False:
            make_dir(outDir)

        #initiate variables
        h5FilesDir = opts.h5FilesDir
        featToSamp = opts.featToSamp
        posRates = opts.posRate.split(',')
        self.chunkSize = opts.chunkSize
        self.minChunkSize = opts.minChunkSize
        self.compressSequence = opts.compressSequence
        self.compressTarget = opts.compressTarget

        # identify all h5 files under the given h5FileDir
        h5Files = glob.glob(os.path.join(h5FilesDir, '*.h5'))
        h5Files.sort()

        # retrieve features from an h5 file
        with h5.File(h5Files[0], 'r') as f:
            self.features = [feature.decode() for feature in f['features']]
            firstChrom = f['chrom'][0].decode("utf-8")

        # iterate over positive rates to sample based on them
        for posRate in posRates:

            # create a directory for each positive rate in outDir
            rateDir = outDir + '/' + posRate + '%'
            if os.path.isdir(rateDir) is False:
                make_dir(rateDir)

            # iterate over features to save datasets in  different ways
            with open(featToSamp, 'r') as f:
                for feat in f:
                    featIdx = self.features.index(feat.rstrip())

                    # create a directory for each feature in rateDir and subdirectories for positive and background data
                    featDir = rateDir + '/' + feat.split('|')[0]
                    if os.path.isdir(rateDir) is False:
                        make_dir(rateDir)
                    posFeatDir = featDir + '/' + 'positive'
                    if os.path.isdir(posFeatDir) is False:
                        make_dir(posFeatDir)
                    backFeatDir = featDir + '/' + 'background'
                    if os.path.isdir(backFeatDir) is False:
                        make_dir(backFeatDir)
                    fullFeatDir = featDir + '/' + 'full'
                    # make copies of hf files in full output directory
                    shutil.copytree(h5FilesDir, fullFeatDir)

                    preChrom = firstChrom
                    preChunkPos, curChunkPos = None, None
                    nSampsPos = 0
                    preChunkBack, curChunkBack = None, None
                    nSampsBack = 0
                    # iterate over each h5 file do sampling in 3 ways
                    for h5File in h5Files:
                        fh = h5.File(h5File, 'r')

                        self.chrom = fh['chrom'][0].decode("utf-8")
                        if self.chrom != preChrom:
                            self.save(preChunkPos, curChunkPos, posFeatDir)
                            self.save(preChunkBack, curChunkBack, backFeatDir)

                            preChunkPos, curChunkPos = None, None
                            nSampsPos = 0
                            preChunkBack, curChunkBack = None, None
                            nSampsBack = 0

                        preChrom = self.chrom

                        targetsUnpacked = np.unpackbits(fh['targets'], axis=-1).astype(float)
                        self.targets = targetsUnpacked[:, :len(self.features)]
                        targetCol = self.targets[:, featIdx]
                        positiveIdx = np.nonzero(targetCol)[0].tolist()
                        nSelectedSamp = round(len(positiveIdx) * float(posRate) / 100)
                        sampleIdx = random.sample(positiveIdx, nSelectedSamp)

                        self.seqLen = fh['sequence_length'][()]
                        sequenceUnpacked = np.unpackbits(fh['sequence'], axis=-2).astype(float)
                        self.sequence = sequenceUnpacked[:, :self.seqLen, :]
                        self.coor = fh['coor'][:]
                        fh.close()

                        # get positive indices and keep %posRate of them and save the h5 files
                        preChunkPos, curChunkPos, nSampsPos = \
                            self.add(preChunkPos, curChunkPos, nSampsPos, sampleIdx, posFeatDir)
                        # remove %posRate of positive examples and save the rest in the h5 files
                        preChunkBack, curChunkBack, nSampsBack = \
                            self.add(preChunkBack, curChunkBack, nSampsBack,
                                     [x for x in range(len(self.targets)) if x not in sampleIdx], backFeatDir)

                        # keep %posRate of positive examples as positive lables and the rest as negative and save it in the h5 files
                        fullFeat = fullFeatDir + '/' + h5File.split('/')[-1]
                        fhFull = h5.File(fullFeat, 'r+')

                        featTargets = np.zeros(len(self.targets))
                        featTargets[sampleIdx] = 1
                        newTargets = self.targets
                        newTargets[:, featIdx] = featTargets

                        del fhFull['targets']
                        if self.compressTarget:
                            fhFull.create_dataset('targets', dtype='uint8', data=np.packbits(newTargets > 0, axis=1))
                        else:
                            fhFull.create_dataset('targets', dtype='float', data=newTargets)
                        fhFull.close()

    def add(self, preChunk, curChunk, nSamps, samps, dir):
        for sample in samps:

            # a valid sample, save it
            nSamps += 1
            if curChunk is None:
                # initialize a chunk
                curChunk = DataChunk(nSamps - 1, self.seqLen, self.features, self.chunkSize)

                # add to the current chunk
            curChunk.add(self.sequence[sample, :, :], self.targets[sample, :], self.chrom, self.coor[sample, :])
            # update the statistics
            #time = timeit.default_timer()
            #self.datastat.add(binLbl, chrom)
            #logger.debug('Time to add statistic:%s' % (timeit.default_timer() - time))

            if (curChunk.isFull()):
                # save the self.preChunk
                if preChunk is not None:
                    preChunk.saveToH5(dir, compressSeq=self.compressSequence, compressLbl=self.compressTarget)
                    #nChunks += 1
                    #logger.info('{0} chunks in {1} have been saved ...'.format(self.nChunks, chrom))

                preChunk = curChunk
                curChunk = None

        return preChunk, curChunk, nSamps

    def save(self, preChunk, curChunk, dir):
        # save chunk and stat
        if preChunk is not None:
            if curChunk is not None and \
                    curChunk.getSize() < self.minChunkSize:
                # merge with previous chunk
                preChunk.merge(curChunk)
                curChunk = None

            preChunk.saveToH5(dir, compressSeq=self.compressSequence, compressLbl=self.compressTarget)
            #nChunks += 1
            #logger.info('{0} chunks in {1} have been saved ...'.format(self.nChunks, chrom))

        if curChunk is not None:
            curChunk.saveToH5(dir, compressSeq=self.compressSequence, compressLbl=self.compressTarget)
            #nChunks += 1
            #logger.info('{0} chunks in {1} have been saved ...'.format(self.nChunks, chrom))


if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
