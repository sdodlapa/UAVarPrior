'''
Created on Apr 30, 2021

@author: jsun
'''
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .dataset import H5Dataset

class H5DataLoader(DataLoader):
    """
    H5DataLoader provides optionally parallel sampling from HDF5
    data files that contains sequences and targets data. The name of the
    array of sequences and targets data are specified by `sequence_key`
    and `targets_key` respectively. The sequences array should be
    of shape:math:`B \\times L \\times N`, where :math:`B` is
    the sample size, :math:`L` is the sequence length, and :math:`N` is
    the size of the sequence type's alphabet. The shape of the targets array
     will be :math:`B \\times F`, where :math:`F` is the number of features.

    H5DataLoader also supports compressed binary data (using `numpy.packbits`)
    with the `unpackbits` option. To generate compressed binary data, the
    sequences and targets array have to both be binary-valued, and then
    packed in the :math:`L` (sequence length) and `F` (number of features)
    dimensions, respectively.
    For the sequences array, represent unknown bases ("N"s) by binary
    data with all-ones in the encoding - they will be transformed to
    the correct representations in fugep.sequences.Genome when unpacked.
    In addition, to unpack correctly, the length of the packed dimensions,
    i.e. :math:`L` and :math:`F` must be provided in two integer scalars
    named `{sequence_key}_length` and `{targets_key}_length` in the HDF5 file
    if `unpackbits==True`.

    Parameters
    ----------
    filepaths: list
        The list of file paths of the hdf5 files.
    nameOfData: list
        The list of names of datasets to load from hdf5 files
    seed: int
        The seed to use for random sampling
    nWorkers : int, optional
        Default is 1. If greater than 1, use multiple processes to parallelize data
        sampling.
    batchSize : int, optional
        Default is 1. Specify the batch size of the DataLoader.
    shuffle : bool, optional
        Default is True. If False, load the data in the original order.

    """
    def __init__(self,
                 filepaths,
                 nameOfData,
                 collateFunc = None,
                 seed = None,
                 nWorkers = 1,
                 batchSize = 1,
                 shuffle = True):
        args = {
            "batch_size": batchSize,
            "num_workers": nWorkers,
            "pin_memory": True,
            "drop_last": True,
            "sampler": None
        }
        if collateFunc is not None:
            args['collate_fn'] = collateFunc
        super(H5DataLoader, self).__init__(
            H5Dataset(filepaths, nameOfData, 
                       seed = seed,
                       shuffle = shuffle),
            **args)
        
        
        
        