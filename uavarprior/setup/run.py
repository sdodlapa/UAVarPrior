"""
Utilities for loading configurations, instantiating Python objects, and
running operations in _Selene_.

"""
import os
import importlib
import sys
from time import strftime
import types
import torch
import torch.nn as nn
import random
import numpy as np
import inspect

import torchinfo
import torch.multiprocessing as mp

# from tensorflow.keras.models import Model

from . import instantiate
from ..model import loadNnModule
from ..model import loadWrapperModule
from ..model.nn.utils import load_model, add_output_layers, make_dir

from torch.distributed import init_process_group, destroy_process_group
from datetime import timedelta
import logging
from typing import Dict, Any
import torch

from uavarprior.models import get_model
# Commented out missing imports to fix compatibility issues
# Original import: from uavarprior.data import get_dataset, get_dataloader
# Fixed import: changed training to train and Trainer to StandardSGDTrainer
from uavarprior.train import StandardSGDTrainer

logger = logging.getLogger(__name__)

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank,
                       world_size=world_size,
                       timeout=timedelta(minutes=30))
    torch.cuda.set_device(rank)

def class_instantiate(classobj):
    """Not used currently, but might be useful later for recursive
    class instantiation
    """
    for attr, obj in classobj.__dict__.items():
        is_module = getattr(obj, '__module__', None)
        if is_module and "uavarprior" in is_module and attr != "model":
            class_instantiate(obj)
    classobj.__init__(**classobj.__dict__)


def module_from_file(path):
    """
    Load a module created based on a Python file path.

    Parameters
    ----------
    path : str
        Path to the model architecture file.

    Returns
    -------
    The loaded module

    """
    parent_path, module_file = os.path.split(path)
    loader = importlib.machinery.SourceFileLoader(
        module_file[:-3], path)
    module = types.ModuleType(loader.name)
    loader.exec_module(module)
    return module


def module_from_dir(path):
    """
    This method expects that you pass in the path to a valid Python module,
    where the `__init__.py` file already imports the model class,
    `criterion`, and `get_optimizer` methods from the appropriate file
    (e.g. `__init__.py` contains the line `from <model_class_file> import
    <ModelClass>`).

    Parameters
    ----------
    path : str
        Path to the Python module containing the model class.

    Returns
    -------
    The loaded module
    """
    parent_path, module_dir = os.path.split(path)
    sys.path.insert(0, parent_path)
    return importlib.import_module(module_dir)

def _getModelInfo(configs, sampler):
    '''
    Assemble model info from the config dictionary
    '''
    modelInfo = configs["model"]
    if not ('classArgs' in modelInfo):
        modelInfo['classArgs'] = dict()
    classArgs = modelInfo['classArgs']
    if not ('sequence_length' in classArgs):
        classArgs['sequence_length'] = sampler.getSequenceLength()
    if not ('n_targets' in classArgs):
        classArgs['n_targets'] = len(sampler.getFeatures())
    
    return modelInfo

def initialize_model(model_configs, train=True, lr=None, configs=None):
    """
    Initialize model (and associated criterion, optimizer)

    Parameters
    ----------
    model_configs : dict
        Model-specific configuration
    train : bool, optional
        Default is True. If `train`, returns the user-specified optimizer
        and optimizer class that can be found within the input model file.
    lr : float or None, optional
        If `train`, a learning rate must be specified. Otherwise, None.

    Returns
    -------
    model, criterion : tuple(torch.nn.Module, torch.nn._Loss) or \
            model, criterion, optim_class, optim_kwargs : \
                tuple(torch.nn.Module, torch.nn._Loss, torch.optim, dict)

        * `torch.nn.Module` - the model architecture
        * `torch.nn._Loss` - the loss function associated with the model
        * `torch.optim` - the optimizer associated with the model
        * `dict` - the optimizer arguments

        The optimizer and its arguments are only returned if `train` is
        True.

    Raises
    ------
    ValueError
        If `train` but the `lr` specified is not a float.

    """
    if model_configs["built"] == 'pytorch':
        model_class_name = model_configs["class"]

        if 'path' in model_configs.keys():
            # load network module from user given file
            import_model_from = model_configs["path"]
            if os.path.isdir(import_model_from):
                module = module_from_dir(import_model_from)
            else:
                module = module_from_file(import_model_from)
        else:
            # load built in network module
            module = loadNnModule(model_class_name)

        model_class = getattr(module, model_class_name)
        ### Get only exprected arguments and ignore extra args
        model_class_expected_argset = set(inspect.getargspec(model_class).args)
        model_class_args = {k: model_configs["classArgs"][k] for k in model_class_expected_argset if k in model_configs["classArgs"]}

        # model = model_class(**model_configs["classArgs"])
        model = model_class(**model_class_args)

        def xavier_initialize(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        if 'xavier_init' in model_configs:
            model.apply(xavier_initialize)


        if "non_strand_specific" in model_configs:
            from uavarprior.model import NonStrandSpecific
            model = NonStrandSpecific(
                model, mode=model_configs["non_strand_specific"])

        # loss function
        if 'criterionArgs' in model_configs:
            criterionArgs = model_configs['criterionArgs']
        else:
            criterionArgs = dict()
        criterion = module.criterion(**criterionArgs)

        # optimizer for training
        optim_class, optim_kwargs = None, None
        if train and isinstance(lr, float):
            optim_class, optim_kwargs = module.get_optimizer(lr)
        elif train:
            raise ValueError("Learning rate must be specified as a float "
                             "but was {0}".format(lr))

    # elif model_configs["built"] == 'tensorflow':
    #     model_class_name = model_configs["class"]
    #     # model_built_name = model_configs["built"]
    #
    #     module = None
    #     if 'path' in model_configs.keys():
    #         # load network module from user given file
    #         import_model_from = model_configs["path"]
    #         if os.path.isdir(import_model_from):
    #             module = module_from_dir(import_model_from)
    #         else:
    #             module = module_from_file(import_model_from)
    #     else:
    #         # load built in network module
    #         module = loadNnModule(model_class_name)
    #
    #     model_class = getattr(module, model_class_name)
    #
    #     model_class_expected_argset = set(inspect.getargspec(model_class).args)
    #     # model_class_args = {k: model_configs["classArgs"][k] for k in model_class_expected_argset if
    #     #                     k in model_configs["classArgs"]}
    #
    #
    #     model_builder = model_class(**model_configs["classArgs"])
    #     # model_builder = model_class(**model_class_args)
    #     dna_wlen = model_configs["dna_wlen"]
    #     dna_inputs = model_builder.inputs(dna_wlen)
    #     stem = model_builder(dna_inputs)
    #     output_names = model_configs['output_names']
    #
    #     # loss function
    #     if 'criterionArgs' in model_configs:
    #         criterionArgs = model_configs['criterionArgs']
    #     else:
    #         criterionArgs = dict()
    #     criterion = module.criterion(**criterionArgs)
    #
    #     outputs = add_output_layers(stem.outputs[0], output_names,
    #                                 loss_fn=criterion)
    #     # from tensorflow.keras.models import Model
    #     model = Model(inputs=stem.inputs, outputs=outputs, name=stem.name)
    #
    #     if "non_strand_specific" in model_configs:
    #         from uavarprior.model import NonStrandSpecific
    #         model = NonStrandSpecific(
    #             model, mode=model_configs["non_strand_specific"])
    #
    #
    #
    #     # optimizer for training
    #     optim_class, optim_kwargs = None, None
    #     if train and isinstance(lr, float):
    #         optim_class, optim_kwargs = module.get_optimizer(lr)
    #     elif train:
    #         raise ValueError("Learning rate must be specified as a float "
    #                          "but was {0}".format(lr))

        # if 'path' in model_configs.keys():
        #     model = load_model(model_configs["path"])
        #
        # model_class_name = model_configs["class"]
        # module = loadNnModule(model_class_name)
        # # TO DO: raise error when non strand specific in configuration variables
        # # if "non_strand_specific" in model_configs:
        # #     from uavarprior.model import NonStrandSpecific
        # #     model = NonStrandSpecific(
        # #         model, mode=model_configs["non_strand_specific"])
        #
        # # loss function
        # if 'criterionArgs' in model_configs:
        #     criterionArgs = model_configs['criterionArgs']
        # else:
        #     criterionArgs = dict()
        # criterion = module.criterion(**criterionArgs)
        #
        # # optimizer for training
        # optim_class, optim_kwargs = None, None
        # if train and isinstance(lr, float):
        #     optim_class, optim_kwargs = module.get_optimizer(lr)
        # elif train:
        #     raise ValueError("Learning rate must be specified as a float "
        #                      "but was {0}".format(lr))

    # construct model wrapper
    if 'plot_grads' in model_configs:
        plot_grads = model_configs['plot_grads']
        if plot_grads:
            gradOutDir = configs['output_dir']
            make_dir(gradOutDir)
    else:
        gradOutDir = None
    if 'rank' in model_configs.keys():
        rank = model_configs['rank']
    else:
        rank = None

    modelWrapper = initializeWrapper(model_configs['wrapper'], 
         mode = 'train', model = model, loss = criterion,
            model_built = model_configs['built'], mult_predictions = model_configs['mult_predictions'],
         optimizerClass = optim_class,  optimizerKwargs = optim_kwargs,
            gradOutDir=gradOutDir, rank=rank
                                     )
    # modelWrapper._model.built = model_configs["built"]
    return modelWrapper

def initializeWrapper(className, mode, model, loss, model_built = 'pytorch', mult_predictions=1, useCuda = None,
          optimizerClass = None, optimizerKwargs = None,
                      gradOutDir=None, rank=None
                      ):
    '''
    Initialize model wrapper
    '''
    wrapperClass = getattr(loadWrapperModule(className), className)
    wrapper = wrapperClass(model, mode = mode, lossCalculator = loss,
                           model_built = model_built, mult_predictions=mult_predictions,
             useCuda = useCuda, optimizerClass = optimizerClass, 
             optimizerKwargs = optimizerKwargs,
                gradOutDir=gradOutDir, rank=rank
                )
    return wrapper
    
def execute(configs):
    """
    Execute operations in _Selene_.

    Parameters
    ----------
    # operations : list(str)
    #     The list of operations to carry out in _Selene_.
    configs : dict or object
        The loaded configurations from a YAML file.
    output_dir : str or None
        The path to the directory where all outputs will be saved.
        If None, this means that an `output_dir` was not specified
        in the top-level configuration keys. `output_dir` must be
        specified in each class's individual configuration wherever
        it is required.

    Returns
    -------
    None
        Executes the operations listed and outputs any files
        to the dirs specified in each operation's configuration.

    Raises
    ------
    ValueError
        If an expected key in configuration is missing.

    """
    model = None
    modelTrainer = None
    output_dir = configs['output_dir']
    for op in configs['ops']:
        if op == "train":
            # ddp_setup(rank, world_size=torch.cuda.device_count())
            # construct sampler
            sampler_info = configs["sampler"]
            if output_dir is not None:
                sampler_info.bind(output_dir=output_dir)
            sampler = instantiate(sampler_info)

            # construct model
            modelInfo = _getModelInfo(configs, sampler)

            # modelInfo['rank'] = rank
            model = initialize_model(modelInfo, train = True,
                                     lr = configs["lr"], configs=configs)

            # ## Create Distributed strategy
            # ddp_setup(rank, world_size=torch.cuda.device_count())
            
            # create trainer
            train_model_info = configs["train_model"]
            train_model_info.bind(model = model, dataSampler = sampler)
            if sampler.getValOfMisInTarget() is not None:
                train_model_info.bind(valOfMisInTarget = sampler.getValOfMisInTarget())
            
            if output_dir is not None:
                train_model_info.bind(outputDir = output_dir)
            if "random_seed" in configs:
                train_model_info.bind(deterministic=True)

            modelTrainer = instantiate(train_model_info)
            
            # train
            modelTrainer.trainAndValidate()
            # destroy_process_group()

        elif op == "evaluate":
            if modelTrainer is not None:
                modelTrainer.evaluate()
            else:
                # construct sampler
                sampler_info = configs["sampler"]
                sampler = instantiate(sampler_info)
                
                # construct model
                modelInfo = _getModelInfo(configs, sampler)
                model = initialize_model(modelInfo, train = False)
                
                # construct evaluator  
                evaluate_model_info = configs["evaluate_model"]
                evaluate_model_info.bind(model = model, dataSampler = sampler)
                if output_dir is not None:
                    evaluate_model_info.bind(outputDir = output_dir)
                if sampler.getValOfMisInTarget() is not None:
                    evaluate_model_info.bind(valOfMisInTarget = sampler.getValOfMisInTarget())
                evaluate_model = instantiate(evaluate_model_info)
                
                # evaluate
                evaluate_model.evaluate()

        elif op == "analyze":
            if not model:
                # TO DO: check argements and allow only the arguments class expects
                model = initialize_model(configs["model"], train = False)
            
            # construct analyzer
            analyze_seqs_info = configs["analyzer"]
            analyze_seqs_info.bind(model = model)
            if output_dir is not None:
                    analyze_seqs_info.bind(outputDir = output_dir)
            analyze_seqs = instantiate(analyze_seqs_info)
            
            if "variant_effect_prediction" in configs:
                analyze_seqs.evaluate()
            elif "in_silico_mutagenesis" in configs:
                ism_info = configs["in_silico_mutagenesis"]
                analyze_seqs.evaluate(**ism_info)
            elif "prediction" in configs:
                predict_info = configs["prediction"]
                analyze_seqs.predict(**predict_info)
            else:
                raise ValueError('The type of analysis needs to be specified. It can '
                                 'either be variant_effect_prediction, in_silico_mutagenesis '
                                 'or prediction')


def parse_configs_and_run(configs: Dict[str, Any]) -> None:
    """Parse configuration and run the specified task.
    
    Args:
        configs: Dictionary containing configuration parameters
    """
    try:
        # Check for CUDA availability and set device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        # Extract key configurations
        data_config = configs.get("data", {})
        model_config = configs.get("model", {})
        training_config = configs.get("training", {})
        
        # Initialize dataset
        logger.info("Initializing dataset...")
        # TODO: Fix missing get_dataset and get_dataloader functions
        # dataset = get_dataset(data_config)
        # dataloader = get_dataloader(dataset, training_config.get("batch_size", 32))
        dataset = None  # Placeholder until get_dataset is implemented
        dataloader = None  # Placeholder until get_dataloader is implemented
        
        # Initialize model
        logger.info(f"Initializing model: {model_config.get('name', 'unknown')}")
        model = get_model(model_config)
        model.to(device)
        
        # Setup training
        logger.info("Setting up training...")
        trainer = StandardSGDTrainer(
            model=model,
            dataloader=dataloader,
            config=training_config,
            device=device
        )
        
        # Run training or inference based on mode
        mode = configs.get("mode", "train")
        if mode == "train":
            logger.info("Starting training...")
            trainer.train()
        elif mode == "evaluate":
            logger.info("Starting evaluation...")
            trainer.evaluate()
        elif mode == "predict":
            logger.info("Starting prediction...")
            trainer.predict()
        else:
            raise ValueError(f"Unknown execution mode: {mode}")
            
    except Exception as e:
        logger.error(f"Execution failed: {str(e)}")
        if configs.get("debug", False):
            import traceback
            traceback.print_exc()
        raise
