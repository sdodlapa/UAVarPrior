"""
This module provides the `TrainModel` class and supporting methods.
"""

from abc import ABCMeta
from abc import abstractmethod

import logging
import os
import shutil
from time import strftime
from time import time

import numpy as np
import torch
from torch.optim import lr_scheduler
# from torch.optim.lr_scheduler import  ReduceLROnPlateau, CosineAnnealingWarmRestarts, CyclicLR

from ...utils import initialize_logger
from ...evaluate import PerformanceMetrics
from ..utils import LossTracker

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from ...evaluate.metrics import f1Neg, AUPRC

logger = logging.getLogger("fugep")

def timeit(func):
    import time
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        elapsed = end - start
        print(f'Time taken for {func}: {elapsed:.6f} seconds')
        return result
    return wrapper

def _metricsLogger(name, out_filepath):
    logger = logging.getLogger("{0}".format(name))
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(message)s")
    file_handle = logging.FileHandler(
        os.path.join(out_filepath, "{0}.log".format(name)))
    file_handle.setFormatter(formatter)
    logger.addHandler(file_handle)
    return logger

def get_metrics(metrics_names):
    metrics_dict = dict(roc_auc=roc_auc_score, auprc=AUPRC,
                    precision=average_precision_score,
                    f1_pos=f1_score, f1_neg=f1Neg, accuracy=accuracy_score,
                    recall=recall_score,
                    confusion_TnFpFnTp=confusion_matrix)
    metrics = {}
    for name in metrics_names:
        try:
            metrics[name] = metrics_dict[name]
        except:
            logger.info("%s metric is not calculated" % name)
    return metrics


class SGDTrainer(metaclass = ABCMeta):
    """
    The base class that trains a model for classification

    TrainModel saves a checkpoint model (overwriting it after
    `nStepsCheckpoint`) as well as a best-performing model
    (overwriting it after `nStepsStatReport` if the latest
    validation performance is better than the previous best-performing
    model) to `outputDir`.

    TrainModel also outputs 2 files that can be used to monitor training
    as Fugep runs: `train.log` (training loss) and
    `validation.log` (validation loss & average
    ROC AUC). The columns in these files can be used to quickly visualize
    training history (e.g. you can use `matplotlib`, `plt.plot(auc_list)`)
    and see, for example, whether the model is still improving, if there are
    signs of overfitting, etc.

    Parameters
    ----------
    model : torch.nn.Module
        The model to train.
    dataSampler : fugep.samplers.Sampler
        The example generator.
    lossCalculator : torch.nn._Loss
        The loss function to optimize.
    optimizerClass : torch.optim.Optimizer
        The optimizer to minimize loss with.
    optimizerKwargs : dict
        The dictionary of keyword arguments to pass to the optimizer's
        constructor.
    batchSize : int
        Specify the batch size to process examples. Should be a power of 2.
    maxSteps : int
        The maximum number of mini-batches to iterate over.
    nStepsStatReport : int
        The frequency with which to report summary statistics. You can
        set this value to be equivalent to a training epoch
        (`n_steps * batchSize`) being the total number of samples
        seen by the model so far. Selene evaluates the model on the validation
        dataset every `nStepsStatReport` and, if the model obtains
        the best performance so far (based on the user-specified loss function),
        Selene saves the model state to a file called `best_model.pth.tar` in
        `outputDir`.
    outputDir : str
        The output directory to save model checkpoints and logs in.
    nStepsCheckpoint : int or None, optional
        Default is 1000. If None, set to the same value as
        `nStepsStatReport`
    nStepsStartCheckpoint : int or None, optional
        Default is None. The number of steps after which Selene will
        continually save new checkpoint model weights files
        (`checkpoint-<TIMESTAMP>.pth.tar`) every
        `nStepsCheckpoint`. Before this point,
        the file `checkpoint.pth.tar` is overwritten every
        `nStepsCheckpoint` to limit the memory requirements.
    nValidationSamples : int or None, optional
        Default is `None`. Specify the number of validation samples in the
        validation set. If `nValidationSamples` is `None` and the data sampler
        used is the `fugep.samplers.IntervalsSampler` or
        `fugep.samplers.RandomSampler`, we will retrieve 32000
        validation samples. If `None` and using
        `fugep.samplers.MultiSampler`, we will use all
        available validation samples from the appropriate data file.
    nTestSamples : int or None, optional
        Default is `None`. Specify the number of test samples in the test set.
        If `nTestSamples` is `None` and

            - the sampler you specified has no test partition, you should not
              specify `evaluate` as one of the operations in the `ops` list.
              That is, Selene will not automatically evaluate your trained
              model on a test dataset, because the sampler you are using does
              not have any test data.
            - the sampler you use is of type `fugep.samplers.OnlineSampler`
              (and the test partition exists), we will retrieve 640000 test
              samples.
            - the sampler you use is of type
              `fugep.samplers.MultiSampler` (and the test partition
              exists), we will use all the test samples available in the
              appropriate data file.

    nCpuThreads : int, optional
        Default is 1. Sets the number of OpenMP threads used for parallelizing
        CPU operations.
    useCuda : bool, optional
        Default is `False`. Specify whether a CUDA-enabled GPU is available
        for torch to use during training.
    dataParallel : bool, optional
        Default is `False`. Specify whether multiple GPUs are available
        for torch to use during training.
    loggingVerbosity : {0, 1, 2}, optional
        Default is 2. Set the logging verbosity level.

            * 0 - Only warnings will be logged.
            * 1 - Information and warnings will be logged.
            * 2 - Debug messages, information, and warnings will all be\
                  logged.

    checkpoint_resume : str or None, optional
        Default is `None`. If `checkpoint_resume` is not None, it should be the
        path to a model file generated by `torch.save` that can now be read
        using `torch.load`.
    useScheduler : bool, optional
        Default is `True`. If `True`, learning rate scheduler is used to
        reduce learning rate on plateau. PyTorch ReduceLROnPlateau scheduler 
        with patience=16 and factor=0.8 is used.

    Attributes
    ----------
    model : torch.nn.Module
        The model to train.
    sampler : fugep.samplers.Sampler
        The example generator.
    batchSize : int
        The size of the mini-batch to use during training.
    maxSteps : int
        The maximum number of mini-batches to iterate over.
    nStepsStatReport : int
        The frequency with which to report summary statistics.
    nStepsCheckpoint : int
        The frequency with which to save a model checkpoint.
    useCuda : bool
        If `True`, use a CUDA-enabled GPU. If `False`, use the CPU.
    dataParallel : bool
        Whether to use multiple GPUs or not.
    outputDir : str
        The directory to save model checkpoints and logs.
    
    Note
    ----------
    Adapted from Selene's TrainModel
    
    """

    def __init__(self,
                 model,
                 dataSampler,
                 outputDir,
                 maxNSteps, # TODO: default None, use early stopping
                 lossCalculator = None, # if None, lossCalculator is set to model directly
                 optimizerClass = None, # if None, optimizer is set to model directly
                 optimizerKwargs = None,
                 batchSize = 64,
                 nStepsStatReport = 100,
                 nStepsCheckpoint = 1000,
                 nStepsStartCheckpoint = None,
                 nMinMinorsReport = 10,
                 nValidationSamples = None,
                 nTestSamples = None,
                 nCpuThreads = 1,
                 useCuda = False,
                 dataParallel = False,
                 loggingVerbosity = 2,
                 preloadValData = False,
                 preloadTestData = False,
                 gradOutDir=None,
                 metrics = None,
                 useScheduler = True,
                 schedulerName=None,
                 changeOptim=False,
                 deterministic=False,
                 valOfMisInTarget = None):
        """
        Constructs a new `TrainModel` object.
        """
        self.model = model
        self.optimizerClass = optimizerClass
        self.optimizerKwargs = optimizerKwargs
        if lossCalculator is not None:
            self.model.setLossCalculator(lossCalculator)
        if self.optimizerClass is not None:
            self.model.setOptimizer(self.optimizerClass, self.optimizerKwargs)
            logger.info("Optimizer: {}".format(self.model._optimizer))

        self.changeOptim = changeOptim
        self.sampler = dataSampler
        self.batchSize = batchSize
        self.maxNSteps = maxNSteps
        self.nStepsStatReport = nStepsStatReport
        self.nStepsCheckpoint = None

        if not nStepsCheckpoint:
            self.nStepsCheckpoint = nStepsStatReport
        else:
            self.nStepsCheckpoint = nStepsCheckpoint

        self._nStepsStartCheckpoint = nStepsStartCheckpoint

        logger.info("Training parameters set: batch size {0}, "
                    "number of steps per 'epoch': {1}, "
                    "maximum number of steps: {2}".format(
                        self.batchSize,
                        self.nStepsStatReport,
                        self.maxNSteps))

        torch.set_num_threads(nCpuThreads)

        self.useCuda = useCuda
        self.dataParallel = dataParallel

        if self.useCuda:
            self.model.toUseCuda()
            logger.debug("Set modules to use CUDA")

        if self.dataParallel:
            self.model.toDataParallel()
            logger.debug("Wrapped model in DistributedDataParallel")

        os.makedirs(outputDir, exist_ok = True)
        self.outputDir = outputDir

        initialize_logger(
            os.path.join(self.outputDir, "fugep.log"),
            verbosity = loggingVerbosity)
        
        self._preloadValData = preloadValData
        self._preloadTestData = preloadTestData
        self._gradOutDir = gradOutDir

        self._nMinMinorsReport = nMinMinorsReport
        self._metrics = get_metrics(metrics)
        self._nValidationSamples = nValidationSamples
        self._nTestSamples = nTestSamples
        self._useScheduler = useScheduler
        self._valOfMisInTarget = valOfMisInTarget
        self.schedulerName = schedulerName

        # self.scheduler_dict = {
        #     'CyclicLR': lr_scheduler.CyclicLR(self.model.getOptimizer(),
        #                               base_lr=1e-6, max_lr=0.1,
        #                               step_size_up=2, step_size_down=8,
        #                               mode='triangular2'),
        #     'ReduceLROnPlateau': lr_scheduler.ReduceLROnPlateau(
        #                          self.model.getOptimizer(), 'min',
        #                          patience=2, factor=0.7),
        #     'CosineAnnealingWarmRestarts': lr_scheduler.CosineAnnealingWarmRestarts(
        #                         self.model.getOptimizer(), T_0=20,
        #                         T_mult=2, eta_min=1e-6),
        #     'SequentialLR': lr_scheduler.SequentialLR,
        #     'StepLR': lr_scheduler.StepLR,
        # }

        self._initTrain()
        self._initValidate()
        if "test" in self.sampler.modes:
            self._initTest()

        if deterministic:
            logger.info("Setting deterministic = True for reproducibility.")
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
        
    def _initTrain(self):
        self._startStep = 0
        self._trainLogger = _metricsLogger(
                "train", self.outputDir)
        self._trainLogger.info("loss")
        # initialize learning rate scheduler only if optimizer API exists
        if self._useScheduler and hasattr(self.model, 'getOptimizer'):
            self.scheduler = lr_scheduler.ReduceLROnPlateau(
                                 self.model.getOptimizer(), 'min',
                                 patience=2, factor=0.7)
        else:
            self.scheduler = None
        self._timePerStep = []
        self._trainLoss = LossTracker()


    def _initValidate(self, needMetric = True):
        self._minLoss = float("inf")
        if self._preloadValData:
            self._createValidationSet(nSamples = self._nValidationSamples)
        
        if self._metrics is not None and needMetric:
            self._validationMetrics = PerformanceMetrics(
                self.sampler.getFeatureByIndex,
                nMinMinorsReport = self._nMinMinorsReport,
                metrics = self._metrics, valOfMisInTarget = self._valOfMisInTarget, isValidate=True)
        
        self._validationLogger = _metricsLogger(
                "validation", self.outputDir)
        self._validationLogger.info("\t".join(["loss"] +
            sorted([x for x in self._validationMetrics.metrics.keys()])))


    def _initTest(self):
        self._testData = None
        if self._preloadTestData:
            self.createTestSet()
        self._testMetrics = PerformanceMetrics(
            self.sampler.getFeatureByIndex,
            nMinMinorsReport = self._nMinMinorsReport,
            metrics = self._metrics, valOfMisInTarget = self._valOfMisInTarget)


    def _createValidationSet(self, nSamples = None):
        """
        Generates the set of validation examples.

        Parameters
        ----------
        nSamples : int or None, optional
            Default is `None`. The size of the validation set. If `None`,
            will use all validation examples in the sampler.

        """
        logger.info("Creating validation dataset.")
        tStart = time()
        self._validationData, self._allValidationTargets = \
            self.sampler.getValidationSet(
                self.batchSize, nSamps = nSamples)
        tFinish = time()
        logger.info(("{0} s to load {1} validation examples ({2} validation "
                     "batches) to evaluate after each training step.").format(
                      tFinish - tStart,
                      nSamples if nSamples else len(self._validationData) * self.batchSize,
                      len(self._validationData)))

    def createTestSet(self):
        """
        Loads the set of test samples.
        We do not create the test set in the `TrainModel` object until
        this method is called, so that we avoid having to load it into
        memory until the model has been trained and is ready to be
        evaluated.

        """
        logger.info("Creating test dataset.")
        tStart = time()
        self._testData, self._allTestTargets = \
            self.sampler.getTestSet(
                self.batchSize, nSamps = self._nTestSamples)
        tFinish = time()
        logger.info(("{0} s to load {1} test examples ({2} test batches) "
                     "to evaluate after all training steps.").format(
                      tFinish - tStart,
                      self._nTestSamples if self._nTestSamples else len(self._testData) * self.batchSize,
                      len(self._testData)))
        np.savez_compressed(
            os.path.join(self.outputDir, "test-targets.npz"),
            data = self._allTestTargets)

    # @profile
    # @timeit
    def _getBatchData(self, mode=None):
        """
        Fetches a mini-batch of examples

        Returns
        -------
        dict()
            A dictionary, the data contained are data type specific.
        """
        batchData = self.sampler.sample(
            batchSize = self.batchSize,
            mode = mode)

        return batchData

    def _loadTrainedModel(self, filepath, newClassifier=None, freezeStem=None):
        '''
        load from a pretrained model

        Parameters
        ------------
        filepath : file path to the saved check point
ng
        '''
        model_state = torch.load(
            filepath,
            map_location = lambda storage, location: storage)

        self.model.init(model_state, newClassifier=newClassifier, freezeStem=freezeStem)





    def _loadCheckpoint(self, filepath, loadComp = True):
        '''
        load from a checkpoint

        Parameters
        ------------
        filepath : file path to the saved check point
        loadComp : indicating the completeness of the loading
            if False, the loaded checkpoint is returned for continuation of
            the loading
        '''
        checkpoint = torch.load(
            filepath,
            map_location = lambda storage, location: storage)
        if "state_dict" not in checkpoint:
            raise ValueError(
                ("'state_dict' not found in file {0} "
                 "loaded with method `torch.load`. Fugep does not support "
                 "continued training of models that were not originally "
                 "trained using Fugep.").format(filepath))

        self.model.init(checkpoint["state_dict"])

        self._startStep = checkpoint["step"]
        if self._startStep >= self.maxNSteps:
            self.maxNSteps += self._startStep

        self._minLoss = checkpoint["min_loss"]
        if not self.changeOptim:
            self.model.initOptim(checkpoint["optimizer"])
        if 'cWeights' in checkpoint:
            self.sampler.setClassWeights(checkpoint['cWeights'])

        if loadComp:
            logger.info(
                ("Resuming from checkpoint: step {0}, min loss {1}").format(
                    self._startStep, self._minLoss))
        else:
            return checkpoint
  
    
    def _checkpoint(self):
        checkpoint_dict = {
            "step": self.step,
            "arch": self.model.__class__.__name__,
            "state_dict": self.model.getStateDict(),
            "min_loss": self._minLoss,
            "optimizer": self.model.getOptimStateDict()
        }
        if self.sampler.getClassWeights() is not None:
            checkpoint_dict['cWeights'] = self.sampler.getClassWeights() 
            
        if self._nStepsStartCheckpoint is not None and \
                self._nStepsStartCheckpoint >= self.step:
            checkpoint_filename = "checkpoint-{0}".format(
                strftime("%m%d%H%M%S"))
            self._saveCheckpoint(
                checkpoint_dict, False, filename=checkpoint_filename)
            logger.debug("Saving checkpoint `{0}.pth.tar`".format(
                checkpoint_filename))
        else:
            self._saveCheckpoint(
                checkpoint_dict, False)
 
            
    def _saveCheckpoint(self, state, isBest, filename = "checkpoint"):
        """
        Saves snapshot of the model state to file. Will save a checkpoint
        with name `<filename>.pth.tar` and, if this is the model's best
        performance so far, will save the state to a `best_model.pth.tar`
        file as well.

        Models are saved in the state dictionary format. This is a more
        stable format compared to saving the whole model (which is another
        option supported by PyTorch). Note that we do save a number of
        additional, Selene-specific parameters in the dictionary
        and that the actual `model.state_dict()` is stored in the `state_dict`
        key of the dictionary loaded by `torch.load`.

        See: https://pytorch.org/docs/stable/notes/serialization.html for more
        information about how models are saved in PyTorch.

        Parameters
        ----------
        state : dict
            Information about the state of the model. Note that this is
            not `model.state_dict()`, but rather, a dictionary containing
            keys that can be used for continued training in Selene
            _in addition_ to a key `state_dict` that contains
            `model.state_dict()`.
        isBest : bool
            Is this the model's best performance so far?
        filename : str, optional
            Default is "checkpoint". Specify the checkpoint filename. Will
            append a file extension to the end of the `filename`
            (e.g. `checkpoint.pth.tar`).

        Returns
        -------
        None

        """
        logger.debug("[TRAIN] {0}: Saving model state to file.".format(
            state["step"]))
        cpFilepath = os.path.join(
            self.outputDir, filename)
        torch.save(state, "{0}.pth.tar".format(cpFilepath))
        if isBest:
            bestFilepath = os.path.join(self.outputDir, "best_model")
            shutil.copyfile("{0}.pth.tar".format(cpFilepath),
                            "{0}.pth.tar".format(bestFilepath))

    
    @abstractmethod
    def trainAndValidate(self):
        """
        Trains the model and measures validation performance.
        """
        raise NotImplementedError()

    def evaluate(self):
        """
        Measures the model test performance.

        Returns
        -------
        dict
            A dictionary, where keys are the names of the loss metrics,
            and the values are the average value for that metric over
            the test set.
        """
        if self._preloadTestData == True:
            loss, predictions = self.model.validate(self._testData)
        else:
            losses = LossTracker()
            predictions = []
            allTargets = []
            
            if self._nTestSamples is None:
                batchData  = self._getBatchData(mode='test')
                while batchData != None:
                    batchPreds, batchLoss, batchnEffTerms = self.model.batchValidate(batchData)
                    predictions.append(batchPreds)
                    allTargets.append(batchData['targets'])
                    losses.add(batchLoss.item(), batchnEffTerms.item())
                    batchData = self._getBatchData(mode='test')
            else:
                count = self.batchSize
                while count < self._nTestSamples:
                    batchData = self._getBatchData(mode='test')
                    batchPreds, batchLoss, batchnEffTerms = self.model.batchValidate(batchData)
                    predictions.append(batchPreds)
                    allTargets.append(batchData['targets'])
                    losses.add(batchLoss.item(), batchnEffTerms.item())
                    count += self.batchSize
                remainder = self.batchSize - (count - self._nTestSamples)
                batchSize = self.batchSize  # Saving original batchSize
                self.batchSize = remainder  # Changing batchSize to remainder
                batchData = self._getBatchData(mode='test')
                batchPreds, batchLoss, batchnEffTerms = self.model.batchValidate(batchData)
                predictions.append(batchPreds)
                allTargets.append(batchData['targets'])
                losses.add(batchLoss.item(), batchnEffTerms.item())
                self.batchSize = batchSize  # Resetting batchSize to original
                
    
            predictions = np.vstack(predictions)
            allTargets = np.vstack(allTargets)
            self._allTestTargets = allTargets
            loss = losses.getAveLoss()
            
        aveScores = self._testMetrics.update(predictions, self._allTestTargets)
            
        aveScores["loss"] = loss
        for name, score in aveScores.items():
            logger.info("test {0}: {1}".format(name, score))
        
        np.savez_compressed(os.path.join(self.outputDir, "test-predictions.npz"),
                            data = predictions)

        perfFilepath = os.path.join(self.outputDir, "test-performance.txt")
        scoresDict = self._testMetrics.write_feature_scores_to_file(
            perfFilepath)

        self._testMetrics.visualize(
            predictions, self._allTestTargets, self.outputDir)

        return (aveScores, scoresDict)





