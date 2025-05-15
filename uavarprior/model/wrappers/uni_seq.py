'''
Created on May 23, 2021

@author: jsun
'''

import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import h5py

from .pred import PredMWrapper
from ...train import LossTracker
from ..utils import loadModel
from ..utils import loadModelFromFile

class UniSeqMWrapper(PredMWrapper):
    '''
    classdocs
    '''
    def __init__(self, model, mode = 'train', lossCalculator = None, model_built = 'pytorch', mult_predictions=1,
             useCuda = False, optimizerClass = None, optimizerKwargs = None,
                 gradOutDir=None, rank=None
                 ):
        '''
        Constructor
        '''
        super(UniSeqMWrapper, self).__init__(model, 
            mode = mode, lossCalculator = lossCalculator,
            model_built = model_built,
            mult_predictions=mult_predictions,
            useCuda = useCuda, optimizerClass = optimizerClass,
            optimizerKwargs = optimizerKwargs,
               gradOutDir = gradOutDir, rank = rank
            )

    # @profile
    def fit(self, batchData, step):
        """
        Fit the model with a batch of data

        Parameters
        ----------
        batchData : dict
            A dictionary that holds the data for training

        Returns
        -------
        float : sum
            The sum of the loss over the batch of the data
        int : nTerms
            The number of terms involved in calculated loss. 
            
        Note
        ------
        The current implementation of this function is one step of gradient update.
        Future implementation can include a nStep argument to support multiple
        steps update or even train to the end (until no improvement over 
        the input batch of data)
        """
        self._model.train()
        inputs = torch.Tensor(batchData['sequence'])
        targets = torch.Tensor(batchData['targets'])
        weights = None
        if 'weights' in batchData:
            weights = torch.Tensor(batchData['weights'])

        if self._useCuda:
            inputs = inputs.cuda()
            targets = targets.cuda()
            if torch.is_tensor(weights):
                weights = weights.cuda() # Added by Sanjeev 07/20/2021

        predictions = self._model(inputs.transpose(1, 2))
        # aveLoss, sumOfLoss, nEffTerms = \
        #     self._lossCalculator(predictions, targets)
        aveLoss, batchLoss, sumOfLoss, nEffTerms = self._lossCalculator(predictions,
                                                        targets, weight = weights)

        self._optimizer.zero_grad()
        # aveLoss.backward()
        batchLoss.backward()
        if self.gradOutDir and step > 0 and step % 1000 == 0:
            self.plot_grads(step)
        if self.gradOutDir and step > 0 and step % 1000 == 0:
            self.save_grads(step)
        self._optimizer.step()
        
        return (sumOfLoss.item(), nEffTerms.item())

    # @profile
    def validate(self, dataInBatches):
        """
        Validate the model with a batch of data

        Parameters
        ----------
        dataInBatches : []
            A list of dictionaries that hold data in batches for the validating

        Returns
        -------
        float : 
            The average loss over the batch of the data
        nArray :
            The prediction
        """
        self._model.eval()

        batchLosses = LossTracker()
        allPreds = []
        count = 0
        for batchData in dataInBatches:
            inputs = torch.Tensor(batchData['sequence'])
            targets = torch.Tensor(batchData['targets'])
            weights = None
            if 'weights' in batchData:
                weights = torch.Tensor(batchData['weights'])

            if self._useCuda:
                inputs = inputs.cuda()
                targets = targets.cuda()
                if torch.is_tensor(weights):
                    weights = weights.cuda()  # Added by Sanjeev 07/21/2021

            with torch.no_grad():
                predictions = self._model(inputs.transpose(1, 2))
                _, _, sumOfLoss, nEffTerms =\
                    self._lossCalculator(predictions, targets, weight = weights)

                allPreds.append(
                    predictions.data.cpu().numpy())
                batchLosses.add(sumOfLoss.item(), nEffTerms.item())
            batchSize = inputs.shape[0]
            count+= batchSize
            if (count // batchSize) % 1000 == 0:
                print(f"Number of samples evaluated: {count}")
        print(f"Total number of samples evaluated: {count}")

        
        allPreds = np.vstack(allPreds)
        return batchLosses.getAveLoss(), allPreds

    # @profile
    def batchValidate(self, batchData):
        """
        Validate the model with a batch of data

        Parameters
        ----------
        batchData : {}
            A dictionary that hold data for one batch

        Returns
        -------
        float :
            The average loss over the batch of the data
        nArray :
            The prediction
        """
        inputs = torch.Tensor(batchData['sequence'])
        targets = torch.Tensor(batchData['targets'])
        weights = None
        if 'weights' in batchData:
            weights = torch.Tensor(batchData['weights'])

        if self._useCuda:
            inputs = inputs.cuda()
            targets = targets.cuda()
            if weights != None:
                weights = weights.cuda()  # Added by Sanjeev 07/21/2021

        with torch.no_grad():
            predictions = self._model(inputs.transpose(1, 2))
            aveLoss, batchLoss, sumOfLoss, nEffTerms = \
                self._lossCalculator(predictions, targets, weight=weights)



        return predictions.data.cpu().numpy().astype(np.float16), sumOfLoss, nEffTerms



    def predict(self, dataInBatches):
        """
        Apply the model to make prediction for a batch of data

        Parameters
        ----------
        batchData : []
            A list of dictionaries that hold data in batches for the validating

        Returns
        -------
        nArray :
            The prediction
        """
        if self._model_built == 'pytorch':
            self._model.eval()
        
            allPreds = []
            for batchData in dataInBatches:
                inputs = torch.Tensor(batchData['sequence'])

                if self._useCuda:
                    inputs = inputs.cuda()
                with torch.no_grad():
                    predictions = self._model(inputs.transpose(1, 2))
                    allPreds.append(predictions.data.cpu().numpy())
            allPreds = np.vstack(allPreds)

        elif self._model_built == 'tensorflow':
            allPreds = []
            for batchData in dataInBatches:
                inputs = batchData['sequence']
                # Uses Tensorflow model default predict function
                predictions = self._model.predict(inputs, verbose=0)
                predictions = np.array(predictions)
                predictions = predictions.reshape((predictions.shape[0],
                                                   predictions.shape[1]))
                predictions = predictions.T
                allPreds.append(predictions)
            allPreds = np.vstack(allPreds)
        
        return allPreds

    def predict_mult(self, dataInBatches):
        """
        Apply the model to make prediction for a batch of data

        Parameters
        ----------
        batchData : []
            A list of dictionaries that hold data in batches for the validating

        Returns
        -------
        nArray :
            The prediction
        """
        num_pred = self._mult_predictions
        if self._model_built == 'pytorch':
            if num_pred > 1:
                self._model.train()
            else:
                self._model.eval()

            for batchData in dataInBatches:
                inputs = torch.Tensor(batchData['sequence'])
                if self._useCuda:
                    inputs = inputs.cuda()
                preds = []
                for n in range(num_pred):
                    with torch.no_grad():
                        predictions = self._model(inputs.transpose(1, 2))
                        # allPreds.append(predictions.data.cpu().numpy())
                        preds.append(predictions.data.cpu().numpy())
            preds = np.array(preds)


        elif self._model_built == 'tensorflow':
            # allPreds = []
            for batchData in dataInBatches:
                preds = []
                inputs = batchData['sequence']
                # Uses Tensorflow model default predict function
                for n in range(num_pred):
                    # predictions = self._model.predict(inputs, verbose=0)
                    predictions = self._model(inputs, training=True)
                    predictions = np.array(predictions)
                    predictions = predictions.reshape((predictions.shape[0],
                                                       predictions.shape[1]))
                    predictions = predictions.T
                    preds.append(predictions)
                # allPreds.append(predictions)

            # allPreds = np.vstack(allPreds)
            preds = np.array(preds)

        return preds

    def init(self, stateDict = None, newClassifier=None, freezeStem=None):
        """
        Initialize the model before training or making prediction
        """
        if stateDict is not None:
            self._model = loadModel(stateDict, self._model, newClassifier=newClassifier, freezeStem=freezeStem)
    
    def initFromFile(self, filepath):
        '''
        Initialize the model by a previously trained model saved 
        to a file
        '''
        loadModelFromFile(filepath, self._model)
    
    def save(self, outputDir, modelName = 'model'):
        """
        Save the model
        
        Parameters:
        --------------
        outputDir : str
            The path to the directory where to save the model
        """
        outputPath = os.path.join(outputDir, modelName)
        torch.save(self._model.state_dict(), 
                   "{0}.pth.tar".format(outputPath))
    def save_grads(self, step):
        layers = []
        grads = []
        for name, param in self._model.named_parameters():
            if 'weight' in name:
                # print(name, param.grad.shape)
                layers.append(''.join(name.split('.')[1:3]))
                grads.append(param.grad.cpu())

        with h5py.File(self.gradOutDir+'/grad_of_step'+str(step)+'.h5', 'w') as h5file:
            for i, layer_name in enumerate(layers):
                group = h5file.create_group(layer_name)
                group.create_dataset('weights', data=grads[i])

    def plot_grads(self, step):
        ave_grads = []
        layers = []
        for name, param in self._model.named_parameters():
            if (param.requires_grad) and ("bias" not in name):
                layers.append(''.join(name.split('.')[1:3]))
                ave_grads.append(param.grad.abs().mean().cpu())
        plt.plot(ave_grads, alpha=0.3, color="b")
        plt.hlines(0, 0, len(ave_grads) + 1, linewidth=1, color="k")
        plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
        plt.xlim(xmin=0, xmax=len(ave_grads))
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title("Gradient flow")
        plt.grid(True)
        plt.savefig(self.gradOutDir+'/gradflow'+str(step)+'.pdf')