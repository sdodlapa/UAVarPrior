"""
An example model that has double the number of convolutional layers
that DeepSEA (Zhou & Troyanskaya, 2015) has. Otherwise, the architecture
is identical to DeepSEA.

We make no claims about the performance of this model. It is being stored
in `utils` so it can be easily loaded in the Jupyter notebook tutorials
for Selene, and may be removed in the future.

When making a model architecture file of your own, please review this
file in its entirety. In addition to the model class, Selene expects
that `criterion` and `get_optimizer(lr)` are also specified in this file.
"""
import numpy as np
import torch
import torch.nn as nn


class DeepSVSD(nn.Module):
    """
    Deep support vector sequence description

    Parameters
    ----------
    sequence_length : int
        The length of the sequences on which the model trains and and makes
        predictions.
    n_targets : int
        The number of targets (classes) to predict.

    Attributes
    ----------
    conv_net : torch.nn.Sequential
        The convolutional neural network component of the model.
    fully_con : torch.nn.Sequential
        The fully connected neural network component of the model
    classifier : torch.nn.Sequential
        The linear classifier and sigmoid transformation components of the
        model.

    """

    def __init__(self, sequence_length, n_targets):
        super(DeepSVSD, self).__init__()
        conv_kernel_size = 8
        pool_kernel_size = 4

        reduce_by = 2 * (conv_kernel_size - 1)
        pool_kernel_size_f = float(pool_kernel_size)
        self._n_channels = int(
            np.floor(
                (np.floor(
                    (sequence_length - reduce_by) / pool_kernel_size_f)
                 - reduce_by) / pool_kernel_size_f)
            - reduce_by)

        self.conv_net = nn.Sequential(
            nn.Conv1d(4, 320, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(320, 320, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(
                kernel_size=pool_kernel_size, stride=pool_kernel_size),
            nn.BatchNorm1d(320),

            nn.Conv1d(320, 480, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(480, 480, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(
                kernel_size=pool_kernel_size, stride=pool_kernel_size),
            nn.BatchNorm1d(480),
            nn.Dropout(p=0.2),

            nn.Conv1d(480, 960, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(960, 960, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(960),
            nn.Dropout(p=0.2))

        self.fully_con = nn.Sequential(
            nn.Linear(960 * self._n_channels, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Dropout(p=0.2),

            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.2),

            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256))

        self.classifier = nn.Sequential(
            # nn.Linear(256, n_targets),
            # nn.ReLU(inplace=True),
            # nn.BatchNorm1d(n_targets),
            nn.Linear(256, n_targets),
            nn.Sigmoid())

    def forward(self, x):
        """
        Forward propagation of a batch.
        """
        out = self.conv_net(x)
        reshape_out = out.view(out.size(0), 960 * self._n_channels)
        output = self.fully_con(reshape_out)
        predict = self.classifier(output)
        return output, predict


def criterion(ocl: True):
    if ocl:
        return OneClassLoss()
    else:
        return nn.BCELoss()


def get_optimizer(lr):
    """
    Specify an optimizer and its parameters.

    Returns
    -------
    tuple(torch.optim.Optimizer, dict)
        The optimizer class and the dictionary of kwargs that should
        be passed in to the optimizer constructor.

    """
    return (torch.optim.SGD,
            {"lr": lr, "weight_decay": 1e-6, "momentum": 0.9})


class OneClassLoss(nn.Module):
    def __init__(self, weight=None):
        super(OneClassLoss, self).__init__()
        # self.weight = weight.cuda()

    def forward(self, dist, R, objective, nu):

        if objective == 'soft-boundary':
            scores = dist - R ** 2
            loss = R ** 2 + (1 / nu) * torch.mean(torch.zeros_like(scores), scores)
        else:
            loss = torch.mean(dist)
        return loss
