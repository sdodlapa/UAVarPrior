"""CpG models.

Provides models trained with observed neighboring methylation states of
multiple cells.
"""

from __future__ import division
from __future__ import print_function

from os import path as pt

import inspect
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers as kl
from tensorflow.keras import regularizers as kr
from tensorflow.keras import models as km
from tensorflow.keras.layers import concatenate
from tensorflow.keras.optimizers import Adam

from .utils import Model
from .utils import get_from_module
from ...train import weightedBCELoss



def criterion():
    """
    The criterion the model aims to minimize.
    """
#     return nn.BCELoss()
    return tf.keras.losses.BinaryCrossentropy
    # return 'binary_crossentropy'

def get_optimizer(lr):
    """
    The optimizer and the parameters with which to initialize the optimizer.
    At a later time, we initialize the optimizer by also passing in the model
    parameters (`model.parameters()`). We cannot initialize the optimizer
    until the model has been initialized.
    """
    return (tf.keras.optimizers.Adam,
            {"lr": lr})


class ScaledSigmoid(kl.Layer):
    """Scaled sigmoid activation function.

    Scales the maximum of the sigmoid function from one to the provided value.

    Parameters
    ----------
    scaling: float
        Maximum of sigmoid function.
    """

    def __init__(self, scaling=1.0, **kwargs):
        self.supports_masking = True
        self.scaling = scaling
        super(ScaledSigmoid, self).__init__(**kwargs)

    def call(self, x, mask=None):
        return K.sigmoid(x) * self.scaling

    def get_config(self):
        config = {'scaling': self.scaling}
        base_config = super(ScaledSigmoid, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


CUSTOM_OBJECTS = {'ScaledSigmoid': ScaledSigmoid}


def search_model_files(dirname):
    """Search model files in given directory.

    Parameters
    ----------
    dirname: str
        Directory name

    Returns
    -------
    Model JSON file and weights if existing, otherwise HDF5 file.  None if no
    model files could be found.
    """

    json_file = pt.join(dirname, 'model.json')
    if pt.isfile(json_file):
        order = ['model_weights.h5', 'model_weights_val.h5',
                 'model_weights_train.h5']
        for name in order:
            filename = pt.join(dirname, name)
            if pt.isfile(filename):
                return [json_file, filename]
    elif pt.isfile(pt.join(dirname, 'model.h5')):
        return pt.join(dirname, 'model.h5')
    else:
        return None


def load_model(model_files, custom_objects=CUSTOM_OBJECTS, log=None):
    """Load Keras model from a list of model files.

    Loads Keras model from list of filenames, e.g. from `search_model_files`.
    `model_files` can be single HDF5 file, or JSON and weights file.

    Parameters
    ----------
    model_file: list
        Input model file names.
    custom_object: dict
        Custom objects for loading models that were trained with custom objects,
        e.g. `ScaledSigmoid`.

    Returns
    -------
    Keras model.
    """
    if not isinstance(model_files, list):
        model_files = [model_files]
    if pt.isdir(model_files[0]):
        model_files = search_model_files(model_files[0])
        if model_files is None:
            raise ValueError('No model found in "%s"!' % model_files[0])
        if log:
            log('Using model files %s' % ' '.join(model_files))
    if pt.splitext(model_files[0])[1] == '.h5':
        model = km.load_model(model_files[0], custom_objects=custom_objects)
    else:
        with open(model_files[0], 'r') as f:
            model = f.read()
        model = km.model_from_json(model, custom_objects=custom_objects)
    if len(model_files) > 1:
        model.load_weights(model_files[1])
    return model


def get_first_conv_layer(layers, get_act=False):
    """Return the first convolutional layers in a stack of layer.

    Parameters
    ----------
    layers: list
        List of Keras layers.
    get_act: bool
        Return the activation layer after the convolutional weight layer.

    Returns
    -------
    Keras layer
        Convolutional layer or tuple of convolutional layer and activation layer
        if `get_act=True`.
    """
    conv_layer = None
    act_layer = None
    for layer in layers:
        if isinstance(layer, kl.Conv1D) and layer.input_shape[-1] == 4:
            conv_layer = layer
            if not get_act:
                break
        elif conv_layer and isinstance(layer, kl.Activation):
            act_layer = layer
            break
    if not conv_layer:
        raise ValueError('Convolutional layer not found')
    if get_act:
        if not act_layer:
            raise ValueError('Activation layer not found')
        return (conv_layer, act_layer)
    else:
        return conv_layer


def save_model(model, model_file, weights_file=None):
    """Save Keras model to file.

    If `model_file` ends with '.h5', saves model description and model weights
    in HDF5 file. Otherwise, saves JSON model description in `model_file`
    and model weights in `weights_file` if provided.

    Parameters
    ----------
    model
        Keras model.
    model_file: str
        Output file.
    weights_file: str
        Weights file.
    """

    if pt.splitext(model_file)[1] == '.h5':
        model.save(model_file)
    else:
        with open(model_file, 'w') as f:
            f.write(model.to_json())
    if weights_file is not None:
        model.save_weights(weights_file, overwrite=True)




class MetNet(Model):
    """Abstract class of a CpG model."""

    def __init__(self, *args, **kwargs):
        super(MetNet, self).__init__(*args, **kwargs)
        self.scope = 'cpg'

    def inputs(self, cpg_wlen, replicate_names):
        inputs = []
        shape = (len(replicate_names), cpg_wlen)
        inputs.append(kl.Input(shape=shape, name='cpg/state'))
        inputs.append(kl.Input(shape=shape, name='cpg/dist'))
        return inputs

    def _merge_inputs(self, inputs):
        return concatenate(inputs, axis=2)


class MetRnnL1(MetNet):
    """Bidirectional GRU with one layer.

    .. code::

        Parameters: 810,000
        Specification: fc[256]_bgru[256]_do
    """

    def __init__(self, act_replicate='relu', *args, **kwargs):
        super(MetRnnL1, self).__init__(*args, **kwargs)
        self.act_replicate = act_replicate

    def _replicate_model(self, input):
        kernel_regularizer = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Dense(256, kernel_initializer=self.init,
                     kernel_regularizer=kernel_regularizer)(input)
        x = kl.Activation(self.act_replicate)(x)

        return km.Model(input, x)

    def __call__(self, inputs):
        x = self._merge_inputs(inputs)

        #         shape = getattr(x, '_keras_shape')
        shape = x.get_shape()
        replicate_model = self._replicate_model(kl.Input(shape=shape[2:]))
        x = kl.TimeDistributed(replicate_model)(x)

        kernel_regularizer = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)
        gru = kl.GRU(256, kernel_regularizer=kernel_regularizer)
        x = kl.Bidirectional(gru)(x)
        x = kl.Dropout(self.dropout)(x)

        return self._build(inputs, x)


class SeqNet(Model):
    """Abstract class of a DNA model."""

    def __init__(self, *args, **kwargs):
        super(SeqNet, self).__init__(*args, **kwargs)
        self.scope = 'dna'

    def inputs(self, dna_wlen):
        return [kl.Input(shape=(dna_wlen, 4), name='dna')]


class SeqCnnL1h128(SeqNet):
    """CNN with one convolutional and one fully-connected layer with 128 units.

    .. code::

        Parameters: 4,100,000
        Specification: conv[128@11]_mp[4]_fc[128]_do
    """

    def __init__(self, nb_hidden=128, *args, **kwargs):
        super(SeqCnnL1h128, self).__init__(*args, **kwargs)
        self.nb_hidden = nb_hidden

    def __call__(self, inputs):
        x = inputs[0]

        kernel_regularizer = kr.L1L2(self.l1_decay, self.l2_decay)
        x = kl.Conv1D(128, 11,
                      kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)

        x = kl.Flatten()(x)

        kernel_regularizer = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Dense(self.nb_hidden,
                     kernel_initializer=self.init,
                     kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(self.dropout)(x)

        return self._build(inputs, x)


class SeqCnnL1h256(SeqCnnL1h128):
    """CNN with one convolutional and one fully-connected layer with 256 units.

    .. code::

        Parameters: 8,100,000
        Specification: conv[128@11]_mp[4]_fc[256]_do
    """

    def __init__(self, *args, **kwargs):
        super(SeqCnnL1h256, self).__init__(*args, **kwargs)
        self.nb_hidden = 256


class SeqCnnL2h128(SeqNet):
    """CNN with two convolutional and one fully-connected layer with 128 units.

    .. code::

        Parameters: 4,100,000
        Specification: conv[128@11]_mp[4]_conv[256@3]_mp[2]_fc[128]_do
    """

    def __init__(self, nb_hidden=128, *args, **kwargs):
        super(SeqCnnL2h128, self).__init__(*args, **kwargs)
        self.nb_hidden = nb_hidden

    def __call__(self, inputs):
        x = inputs[0]

        kernel_regularizer = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(128, 11,
                      kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)
        if self.batch_norm:
            x = kl.BatchNormalization()(x)

        kernel_regularizer = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(256, 3,
                      kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(2)(x)
        if self.batch_norm:
            x = kl.BatchNormalization()(x)

        x = kl.Flatten()(x)

        kernel_regularizer = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Dense(self.nb_hidden,
                     kernel_initializer=self.init,
                     kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(self.dropout)(x)

        return self._build(inputs, x)


class SeqCnnL2h256(SeqCnnL2h128):
    """CNN with two convolutional and one fully-connected layer with 256 units.

    .. code::

        Parameters: 8,100,000
        Specification: conv[128@11]_mp[4]_conv[256@3]_mp[2]_fc[256]_do
    """

    def __init__(self, *args, **kwargs):
        super(SeqCnnL2h256, self).__init__(*args, **kwargs)
        self.nb_hidden = 256


class CnnL2h256(SeqCnnL2h128):
    """CNN with two convolutional and one fully-connected layer with 256 units.

    .. code::

        Parameters: 8,100,000
        Specification: conv[128@11]_mp[4]_conv[256@3]_mp[2]_fc[256]_do
    """

    def __init__(self,  *args, **kwargs):
        super(CnnL2h256, self).__init__(*args, **kwargs)
        self.nb_hidden = 256


class CnnL3h128(SeqNet):
    """CNN with three convolutional and one fully-connected layer with 128 units.

    .. code::

        Parameters: 4,400,000
        Specification: conv[128@11]_mp[4]_conv[256@3]_mp[2]_conv[512@3]_mp[2]_
                       fc[128]_do
    """

    def __init__(self, nb_hidden=128, *args, **kwargs):
        super(CnnL3h128, self).__init__(*args, **kwargs)
        self.nb_hidden = nb_hidden

    def __call__(self, inputs):
        x = inputs[0]

        kernel_regularizer = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(128, 11,
                      kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)

        kernel_regularizer = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(256, 3,
                      kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(2)(x)

        kernel_regularizer = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(512, 3,
                      kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(2)(x)

        x = kl.Flatten()(x)

        kernel_regularizer = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Dense(self.nb_hidden,
                     kernel_initializer=self.init,
                     kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(self.dropout)(x, training=True)

        return self._build(inputs, x)


class CnnL3h256(CnnL3h128):
    """CNN with three convolutional and one fully-connected layer with 256 units.

    .. code::

        Parameters: 8,300,000
        Specification: conv[128@11]_mp[4]_conv[256@3]_mp[2]_conv[512@3]_mp[2]_
                       fc[256]_do
    """

    def __init__(self,  *args, **kwargs):
        super(CnnL3h256, self).__init__(*args, **kwargs)
        self.nb_hidden = 256



class JointNet(Model):
    """Abstract class of a Joint model."""

    def __init__(self, *args, **kwargs):
        super(JointNet, self).__init__(*args, **kwargs)
        self.mode = 'concat'
        self.scope = 'joint'

    def _get_inputs_outputs(self, models):
        inputs = []
        outputs = []
        for model in models:
            inputs.extend(model.inputs)
            outputs.extend(model.outputs)
        return (inputs, outputs)

    def _build(self, models, layers=[]):
        for layer in layers:
            layer._name = '%s/%s' % (self.scope, layer._name)

        inputs, outputs = self._get_inputs_outputs(models)
        x = concatenate(outputs)
        for layer in layers:
            x = layer(x)

        model = km.Model(inputs, x, name=self.name)
        return model


class JointL0(JointNet):
    """Concatenates inputs without trainable layers.

    .. code::

        Parameters: 0
    """

    def __call__(self, models):
        return self._build(models)


class JointL1h512(JointNet):
    """One fully-connected layer with 512 units.

    .. code::

        Parameters: 524,000
        Specification: fc[512]
    """

    def __init__(self, nb_layer=1, nb_hidden=512, *args, **kwargs):
        super(JointL1h512, self).__init__(*args, **kwargs)
        self.nb_layer = nb_layer
        self.nb_hidden = nb_hidden

    def __call__(self, models):
        layers = []
        for layer in range(self.nb_layer):
            kernel_regularizer = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)
            layers.append(kl.Dense(self.nb_hidden,
                                   kernel_initializer=self.init,
                                   kernel_regularizer=kernel_regularizer))
            layers.append(kl.Activation('relu'))
            if self.batch_norm:
                layers.append(kl.BatchNormalization())
            layers.append(kl.Dropout(self.dropout))

        return self._build(models, layers)


class JointL2h512(JointL1h512):
    """Two fully-connected layers with 512 units.

    .. code::

        Parameters: 786,000
        Specification: fc[512]_fc[512]
    """

    def __init__(self, *args, **kwargs):
        super(JointL2h512, self).__init__(*args, **kwargs)
        self.nb_layer = 2


class JointL3h512(JointL1h512):
    """Three fully-connected layers with 512 units.

    .. code::

        Parameters: 1,000,000
        Specification: fc[512]_fc[512]_fc[512]
    """

    def __init__(self, *args, **kwargs):
        super(JointL3h512, self).__init__(*args, **kwargs)
        self.nb_layer = 3


def list_models():
    """Return the name of models in the module."""

    models = dict()
    for name, value in globals().items():
        if inspect.isclass(value) and name.lower().find('model') == -1:
            models[name] = value
    return models


def get(name):
    """Return object from module by its name."""
    return get_from_module(name, globals())
