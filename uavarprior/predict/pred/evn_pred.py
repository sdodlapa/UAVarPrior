'''
Created on May 27, 2021

@author: jsun
'''

from abc import ABCMeta
from abc import abstractmethod

from ...data import Genome
from ..analyzer import Analyzer

class EventPredictor(Analyzer, metaclass = ABCMeta):
    '''
    Predict functional genomic events
    '''

    def __init__(self, model, trainedModelPath, features,
                 outputDir = None, outputFormat = 'tsv',
                 seqLen = None, batchSize = 64, useCuda = False,
                 dataParallel = False, refSeq = Genome,
                 writeMemLimit = 5000, loggingVerbosity = 2):
        '''
        Construct a new object of 'EventPredictor'
        '''
        super(EventPredictor, self).__init__(model = model, 
                 trainedModelPath = trainedModelPath,
                 features = features, 
                 analysis = ['predictions'], 
                 outputDir = outputDir,
                 outputFormat = outputFormat,
                 seqLen = seqLen,
                 mode = 'prediction',
                 batchSize = batchSize, 
                 useCuda = useCuda,
                 dataParallel = dataParallel, 
                 refSeq = refSeq,
                 writeMemLimit = writeMemLimit,
                 loggingVerbosity = loggingVerbosity)
        
        
    @abstractmethod    
    def predict(self, inputData):
        '''
        Make prediction using data in the parameter 'inputData' as input data, 
        which can be either a dictionary or path to a file.
        The content of the dictionary and the format of the file depends on 
        the type of genomic event to predict
        '''
        
        raise NotImplementedError()
    
    
    
    
    