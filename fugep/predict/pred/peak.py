'''
Created on May 28, 2021

@author: jsun
'''

import numpy as np
import os
import logging
import pyfaidx

from .evn_pred import EventPredictor
from ...data import Genome

logger = logging.getLogger("fugep")

class PeakEventPredictor(EventPredictor):
    '''
    classdocs
    '''

    def __init__(self, model, trainedModelPath, features,
                 outputDir = None, outputFormat = 'tsv',
                 seqLen = None, batchSize = 64, useCuda = False,
                 dataParallel = False, refSeq = Genome,
                 writeMemLimit = 5000, loggingVerbosity = 2):
        '''
        Construct a new object of 'EventPredictor'
        
        Parameters
        -----------
        strandIdx : int or None, optional
            Default is None. If the trained model makes strand-specific
            predictions, your input file may include a column with strand
            information (strand must be one of {'+', '-', '.'}). Specify
            the index (0-based) to use it. Otherwise, by default '+' is used.
        '''
        
        super(PeakEventPredictor, self).__init__(model = model, 
                 trainedModelPath = trainedModelPath,
                 features = features, 
                 outputDir = outputDir, 
                 outputFormat = outputFormat,
                 seqLen = seqLen,
                 batchSize = batchSize, 
                 useCuda = useCuda,
                 dataParallel = dataParallel, 
                 refSeq = refSeq,
                 writeMemLimit = writeMemLimit,
                 loggingVerbosity = loggingVerbosity)
        

    def _predictForBedFile(self, bedFilePath, outputFormat = None, 
                           strandIdx = None):
        """
        Get model predictions for sequences specified as genome coordinates
        in a BED file. Coordinates do not need to be the same length as the
        model expected sequence input--predictions will be centered at the
        midpoint of the specified start and end coordinates.

        Parameters
        ----------
        bedFilePath : str
            Input path to the BED file.
        outputFormat : {'tsv', 'hdf5'}, optional
            When not given, self._outputFormat is used. 
            Choose whether to save TSV or HDF5 output files.
            TSV is easier to access (i.e. open with text editor/Excel) and
            quickly peruse, whereas HDF5 files must be accessed through
            specific packages/viewers that support this format (e.g. h5py
            Python package). Choose

                * 'tsv' if your list of sequences is relatively small
                  (:math:`10^4` or less in order of magnitude) and/or your
                  model has a small number of features (<1000).
                * 'hdf5' for anything larger and/or if you would like to
                  access the predictions/scores as a matrix that you can
                  easily filter, apply computations, or use in a subsequent
                  classifier/model. In this case, you may access the matrix
                  using `mat["data"]` after opening the HDF5 file using
                  `mat = h5py.File("<output.h5>", 'r')`. The matrix columns
                  are the features and will match the same ordering as your
                  features .txt file (same as the order your model outputs
                  its predictions) and the matrix rows are the sequences.
                  Note that the row labels (FASTA description/IDs) will be
                  output as a separate .txt file (should match the ordering
                  of the sequences in the input FASTA).

        strandIdx : int or None, optional
            Default is None. If the trained model makes strand-specific
            predictions, your input file may include a column with strand
            information (strand must be one of {'+', '-', '.'}). Specify
            the index (0-based) to use it. Otherwise, by default '+' is used.


        Returns
        -------
        None
            Writes the output to file(s) in `self._outputdir`. Filename will
            match that specified in the bedFilePath.

        """
        _, filename = os.path.split(bedFilePath)
        outputPrefix = '.'.join(filename.split('.')[:-1])
        outputPath = os.path.join(self._outputDir, outputPrefix)

        seqCoords, labels = self._getSeqFromBedFile(bedFilePath, 
                        outputPath, strandIdx = strandIdx)

        reporter = self._initializeReporters(
            outputPath,
            outputSize = len(labels),
            colNamesOfIds = ["index", "chrom", "start", "end", "strand", "contains_unk"],
            outputFormat = outputFormat)[0]
        sequences = None
        batchIds = []
        for i, (label, coords) in enumerate(zip(labels, seqCoords)):
            encoding, containsUnk = self._refSeq.get_encoding_from_coords_check_unk(
                    *coords, pad=True)
            if sequences is None:
                sequences = np.zeros((self._batchSize, *encoding.shape))
            if i and i % self._batchSize == 0:
                preds = self._model.predict([{'sequence': sequences}])
                reporter.handle_batch_predictions(preds, batchIds)
                sequences = np.zeros((self._batchSize, *encoding.shape))
                batchIds = []
            sequences[i % self._batchSize, :, :] = encoding
            batchIds.append(label+(containsUnk,))
            if containsUnk:
                logger.warn(("For region {0}, "
                               "reference sequence contains unknown "
                               "base(s). --will be marked `True` in the "
                               "`contains_unk` column of the .tsv or "
                               "row_labels .txt file.").format(label))

        sequences = sequences[:i % self._batchSize + 1, :, :]
        preds = self._model.predict([{'sequence': sequences}])
        reporter.handle_batch_predictions(preds, batchIds)
        reporter.write_to_file()


    def _predictForFastaFile(self, filepath, outputFormat):
        """
        Get model predictions for sequences in a FASTA file.

        Parameters
        ----------
        filePath : str
            Input path to the FASTA file.
        outputFormat : {'tsv', 'hdf5'}
            Default is 'tsv'. Choose whether to save TSV or HDF5 output files.
            TSV is easier to access (i.e. open with text editor/Excel) and
            quickly peruse, whereas HDF5 files must be accessed through
            specific packages/viewers that support this format (e.g. h5py
            Python package). Choose

                * 'tsv' if your list of sequences is relatively small
                  (:math:`10^4` or less in order of magnitude) and/or your
                  model has a small number of features (<1000).
                * 'hdf5' for anything larger and/or if you would like to
                  access the predictions/scores as a matrix that you can
                  easily filter, apply computations, or use in a subsequent
                  classifier/model. In this case, you may access the matrix
                  using `mat["data"]` after opening the HDF5 file using
                  `mat = h5py.File("<output.h5>", 'r')`. The matrix columns
                  are the features and will match the same ordering as your
                  features .txt file (same as the order your model outputs
                  its predictions) and the matrix rows are the sequences.
                  Note that the row labels (FASTA description/IDs) will be
                  output as a separate .txt file (should match the ordering
                  of the sequences in the input FASTA).

        Returns
        -------
        None
            Writes the output to file(s) in `self._outputDir`.

        """
        _, filename = os.path.split(filepath)
        outputPrefix = '.'.join(filename.split('.')[:-1])

        fastaFile = pyfaidx.Fasta(filepath)
        reporter = self._initialize_reporters(["predictions"],
            os.path.join(self._outputDir, outputPrefix),
            outputFormat = outputFormat, colNamesOfIds = ["index", "name"],
            output_size = len(fastaFile.keys()))[0]
        sequences = np.zeros((self._batchSize, self.sequence_length,
                              len(self._refSeq.BASES_ARR)))
        batchIds = []
        for i, fastaRec in enumerate(fastaFile):
            seq = self._padOrTruncateSeq(str(fastaRec))
            seqEnc = self._refSeq.sequence_to_encoding(seq)

            if i and i > 0 and i % self._batchSize == 0:
                preds = self._model.predict([{'sequences': sequences}])
                sequences = np.zeros((self._batchSize, *seqEnc.shape))
                reporter.handle_batch_predictions(preds, batchIds)
                batchIds = []

            batchIds.append([i, fastaRec.name])
            sequences[i % self._batchSize, :, :] = seqEnc

        sequences = sequences[:i % self._batchSize + 1, :, :]
        preds = self._model.predict([{'sequences': sequences}])
        reporter.handle_batch_predictions(preds, batchIds)

        fastaFile.close()
        reporter.write_to_file()


    def predict(self, inputData = None, outputFormat = None, strandIdx = None):
        """
        Get model predictions for sequences specified as a raw sequence,
        FASTA, or BED file.

        Parameters
        ----------
        inputData : str
            A dictionary contains a single sequence.
        output_dir : str, optional
            Default is None. Output directory to write the model predictions.
            If this is left blank a raw sequence input will be assumed, though
            an output directory is required for FASTA and BED inputs.
        outputFormat : {'tsv', 'hdf5'}, optional
            If left None, self._outputFormat will be used. 
            Choose whether to save TSV or HDF5 output files.
            TSV is easier to access (i.e. open with text editor/Excel) and
            quickly peruse, whereas HDF5 files must be accessed through
            specific packages/viewers that support this format (e.g. h5py
            Python package). Choose

                * 'tsv' if your list of sequences is relatively small
                  (:math:`10^4` or less in order of magnitude) and/or your
                  model has a small number of features (<1000).
                * 'hdf5' for anything larger and/or if you would like to
                  access the predictions/scores as a matrix that you can
                  easily filter, apply computations, or use in a subsequent
                  classifier/model. In this case, you may access the matrix
                  using `mat["data"]` after opening the HDF5 file using
                  `mat = h5py.File("<output.h5>", 'r')`. The matrix columns
                  are the features and will match the same ordering as your
                  features .txt file (same as the order your model outputs
                  its predictions) and the matrix rows are the sequences.
                  Note that the row labels (FASTA description/IDs) will be
                  output as a separate .txt file (should match the ordering
                  of the sequences in the input FASTA).

        strandIdx : int or None, optional
            Default is None. If the trained model makes strand-specific
            predictions, your input BED file may include a column with strand
            information (strand must be one of {'+', '-', '.'}). Specify
            the index (0-based) to use it. Otherwise, by default '+' is used.
            (This parameter is ignored if FASTA file is used as input.)

        Returns
        -------
        None
            Writes the output to file(s) in `output_dir`. Filename will
            match that specified in the filepath. In addition, if any base
            in the given or retrieved sequence is unknown, the row labels .txt file
            or .tsv file will mark this sequence or region as `contains_unk = True`.

        """
        
        if isinstance(inputData, str):
            if not(inputData.endswith('.fa') or inputData.endswith('.fasta') or \
                inputData.endswith('.bed')):
                raise ValueError('Unrecognized input file: {0}'.format(inputData))
            
            if self._outputDir is None:
                raise ValueError('Output directory (outputDir) is not set. '
                         'outputDir has to be set for saving the prediction results.')
            os.makedirs(self._outputDir, exist_ok = True)
            
            if inputData.endswith('.fa') or inputData.endswith('.fasta'):
                self._predictForFastaFile(inputData, outputFormat)
            else:
                self._predictForBedFile(inputData, outputFormat, strandIdx)
        else:
            # a dictionary that contains sequence key is assumed
            sequence = self._padOrTruncateSeq(inputData['sequence'])
            seqEnc = self._refSeq.sequence_to_encoding(sequence)
            seqEnc = np.expand_dims(seqEnc, axis = 0)  # add batch size of 1
            return self._model.predict([{'sequence': seqEnc}])

