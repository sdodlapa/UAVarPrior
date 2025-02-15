'''
Created on May 28, 2021

@author: jsun
'''

from abc import ABCMeta

from ..seq_analyzer import SeqAnalyzer
from ....data import Genome

class ISMSeqAnalyzer(SeqAnalyzer, metaclass = ABCMeta):
    '''
    Base class of sequence analyzers via in-silico mutagenesis
    '''
    
    ISM_COLS = ["pos", "ref", "alt"]

    def __init__(self, analysis, model, trainedModelPath, 
                 features, nMutBase = 1, outputDir = None, 
                 outputFormat = 'tsv', 
                 seqLen = None, batchSize = 64, useCuda = False,
                 dataParallel = False, refSeq = Genome,
                 writeMemLimit = 5000, loggingVerbosity = 2):
        '''
        Construct a new object of 'SeqAnalyzer'
        '''
        super(ISMSeqAnalyzer, self).__init__(model = model, 
                 trainedModelPath = trainedModelPath,
                 features = features, 
                 analysis = analysis, 
                 outputDir = outputDir,
                 outputFormat = outputFormat,
                 seqLen = seqLen,
                 mode = 'ism',
                 batchSize = batchSize, 
                 useCuda = useCuda,
                 dataParallel = dataParallel, 
                 refSeq = refSeq,
                 writeMemLimit = writeMemLimit,
                 loggingVerbosity = loggingVerbosity)

        self._nMutBase = nMutBase


