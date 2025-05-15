"""General-purpose functions."""

from __future__ import division
from __future__ import print_function

import threading
from collections import OrderedDict
import os
import re
import six
from h5py import h5
from six.moves import range
from os import path as pt

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import models as km
from tensorflow.keras import layers as kl
from tensorflow.keras.utils import to_categorical
import numpy as np


EPS = 10e-8
# Constant for missing labels.
CPG_NAN = -1
# Constant for separating output names, e.g. 'cpg/cell'.
OUTPUT_SEP = '/'


def make_dir(dirname):
    """Create directory `dirname` if non-existing.

    Parameters
    ----------
    dirname: str
        Path of directory to be created.

    Returns
    -------
    bool
        `True`, if directory did not exist and was created.
    """
    if os.path.exists(dirname):
        return False
    else:
        os.makedirs(dirname)
        return True


class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """

    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.it)

    def next(self):
        return self.__next__()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe."""

    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))

    return g


def slice_dict(data, idx):
    """Slice elements in dict `data` by `idx`.

    Slices array-like objects in `data` by index `idx`. `data` can be
    tree-like with sub-dicts, where the leafs must be sliceable by `idx`.

    Parameters
    ----------
    data: dict
        dict to be sliced.
    idx: slice
        Slice index.

    Returns
    -------
    dict
        dict with same elements as in `data` with sliced by `idx`.
    """
    if isinstance(data, dict):
        data_sliced = dict()
        for key, value in six.iteritems(data):
            data_sliced[key] = slice_dict(value, idx)
        return data_sliced
    else:
        return data[idx]


def fold_dict(data, nb_level=10 ** 5):
    """Fold dict `data`.

    Turns dictionary keys, e.g. 'level1/level2/level3', into sub-dicts, e.g.
    data['level1']['level2']['level3'].

    Parameters
    ----------
    data: dict
        dict to be folded.
    nb_level: int
        Maximum recursion depth.

    Returns
    -------
    dict
        Folded dict.
    """
    if nb_level <= 0:
        return data

    groups = dict()
    levels = set()
    for key, value in data.items():
        idx = key.find('/')
        if idx > 0:
            level = key[:idx]
            group_dict = groups.setdefault(level, dict())
            group_dict[key[(idx + 1):]] = value
            levels.add(level)
        else:
            groups[key] = value
    for level in levels:
        groups[level] = fold_dict(groups[level], nb_level - 1)
    return groups


def add_to_dict(src, dst):
    """Add `dict `src` to `dict` `dst`

    Adds values in `dict` `src` to `dict` `dst` with same keys but values are
    lists of added values. lists of values in `dst` can be stacked with
    :func:`stack_dict`.  Used for example in `dpcg_eval.py` to stack dicts from
    different batches.

    Example
    -------
    src = dict()
    src['a'] = 1
    src['b'] = {'b1': 10}
    dst = dict()
    add_to_dict(src, dst)
    add_to_dict(src, dst)
    -> dst['a'] = [1, 1]
    -> dst['b'] = {'b1': [10, 10]}
    """
    for key, value in six.iteritems(src):
        if isinstance(value, dict):
            if key not in dst:
                dst[key] = dict()
            add_to_dict(value, dst[key])
        else:
            if key not in dst:
                dst[key] = []
            dst[key].append(value)


def stack_dict(data):
    """Stacks lists of numpy arrays in `dict` `data`."""
    sdata = dict()
    for key, value in six.iteritems(data):
        if isinstance(value, dict):
            sdata[key] = stack_dict(value)
        else:
            fun = np.vstack if value[0].ndim > 1 else np.hstack
            sdata[key] = fun(value)
    return sdata


def linear_weights(length, start=0.1):
    """Create linear-triangle weights.

    Create array `x` of length `length` with linear weights, where the weight is
    highest (one) for the center x[length//2] and lowest (`start` ) at the ends
    x[0] and x[-1].

    Parameters
    ----------
    length: int
        Length of the weight array.
    start: float
        Minimum weights.

    Returns
    -------
    :class:`np.ndarray`
        Array of length `length` with weight.
    """
    weights = np.linspace(start, 1, np.ceil(length / 2))
    tmp = weights
    if length % 2:
        tmp = tmp[:-1]
    weights = np.hstack((weights, tmp[::-1]))
    return weights


def to_list(value):
    """Convert `value` to a list."""
    if not isinstance(value, list) and value is not None:
        value = [value]
    return value


def move_columns_front(frame, columns):
    """Move `columns` of Pandas DataFrame to the front."""
    if not isinstance(columns, list):
        columns = [columns]
    columns = [column for column in columns if column in frame.columns]
    return frame[columns + list(frame.columns[~frame.columns.isin(columns)])]


def get_from_module(identifier, module_params, ignore_case=True):
    """Return object from module.

    Return object with name `identifier` from module with items `module_params`.

    Parameters
    ----------
    identifier: str
        Name of object, e.g. a function, in module.
    module_params: dict
        `dict` of items in module, e.g. `globals()`
    ignore_case: bool
        If `True`, ignore case of `identifier`.

    Returns
    -------
    object
        Object with name `identifier` in module, e.g. a function or class.
    """
    if ignore_case:
        _module_params = dict()
        for key, value in six.iteritems(module_params):
            _module_params[key.lower()] = value
        _identifier = identifier.lower()
    else:
        _module_params = module_params
        _identifier = identifier
    item = _module_params.get(_identifier)
    if not item:
        raise ValueError('Invalid identifier "%s"!' % identifier)
    return item


def format_table_row(values, widths=None, sep=' | '):
    """Format a row with `values` of a table."""
    if widths:
        _values = []
        for value, width in zip(values, widths):
            if value is None:
                value = ''
            _values.append('{0:>{1}s}'.format(value, width))
    return sep.join(_values)


def format_table(table, colwidth=None, precision=2, header=True, sep=' | '):
    """Format a table of values as string.

    Formats a table represented as a `dict` with keys as column headers and
    values as a lists of values in each column.

    Parameters
    ----------
    table: `dict` or `OrderedDict`
        `dict` or `OrderedDict` with keys as column headers and values as lists
        of values in each column.
    precision: int or list of ints
        Precision of floating point values in each column. If `int`, uses same
        precision for all columns, otherwise formats columns with different
        precisions.
    header: bool
        If `True`, print column names.
    sep: str
        Column separator.

    Returns
    -------
    str
        String of formatted table values.
    """

    col_names = list(table.keys())
    if not isinstance(precision, list):
        precision = [precision] * len(col_names)
    col_widths = []
    tot_width = 0
    nb_row = None
    ftable = OrderedDict()
    for col_idx, col_name in enumerate(col_names):
        width = max(len(col_name), precision[col_idx] + 2)
        values = []
        for value in table[col_name]:
            if value is None:
                value = ''
            elif isinstance(value, float):
                value = '{0:.{1}f}'.format(value, precision[col_idx])
            else:
                value = str(value)
            width = max(width, len(value))
            values.append(value)
        ftable[col_name] = values
        col_widths.append(width)
        if not nb_row:
            nb_row = len(values)
        else:
            nb_row = max(nb_row, len(values))
        tot_width += width
    tot_width += len(sep) * (len(col_widths) - 1)
    rows = []
    if header:
        rows.append(format_table_row(col_names, col_widths, sep=sep))
        rows.append('-' * tot_width)
    for row in range(nb_row):
        values = []
        for col_values in six.itervalues(ftable):
            if row < len(col_values):
                values.append(col_values[row])
            else:
                values.append(None)
        rows.append(format_table_row(values, col_widths, sep=sep))
    return '\n'.join(rows)


def filter_regex(values, regexs):
    """Filters list of `values` by list of `regexs`.

    Paramters
    ---------
    values: list
        list of `str` values.
    regexs: list
        list of `str` regexs.

    Returns
    -------
    list
        Sorted `list` of values in `values` that match any regex in `regexs`.
    """
    if not isinstance(values, list):
        values = [values]
    if not isinstance(regexs, list):
        regexs = [regexs]
    filtered = set()
    for value in values:
        for regex in regexs:
            if re.search(regex, value):
                filtered.add(value)
    return sorted(list(filtered))


class ProgressBar(object):
    """Vertical progress bar.

    Unlike the progressbar2 package, logs progress as multiple lines instead of
    single line, which enables printing to a file. Used, for example, in

    Parameters
    ----------
    nb_tot: int
        Maximum value
    logger: function
        Function that takes a `str` and prints it.
    interval: float
        Logging frequency as fraction of one. For example, 0.1 logs every tenth
        value.

    See also
    --------
    dcpg_eval.py and dcpg_filter_act.py.
    """

    def __init__(self, nb_tot, logger=print, interval=0.1):
        if nb_tot <= 0:
            raise ValueError('Total value must be greater than zero!')
        self.nb_tot = nb_tot
        self.logger = logger
        self.interval = interval
        self._value = 0
        self._nb_interval = 0

    def update(self, amount):
        tricker = self._value == 0
        amount = min(amount, self.nb_tot - self._value)
        self._value += amount
        self._nb_interval += amount
        tricker |= self._nb_interval >= int(self.nb_tot * self.interval)
        tricker |= self._value >= self.nb_tot
        if tricker:
            nb_digit = int(np.floor(np.log10(self.nb_tot))) + 1
            msg = '{value:{nb_digit}d}/{nb_tot:d} ({per:3.1f}%)'
            msg = msg.format(value=self._value, nb_digit=nb_digit,
                             nb_tot=self.nb_tot,
                             per=self._value / self.nb_tot * 100)
            self.logger(msg)
            self._nb_interval = 0

    def close(self):
        if self._value < self.nb_tot:
            self.update(self.nb_tot)


def copy_weights(src_model, dst_model):
    """Copy weights from `src_model` to `dst_model`.

    Parameters
    ----------
    src_model
        Keras source model.
    dst_model
        Keras destination model.

    Returns
    -------
    list
        Names of layers that were copied.
    """
    assert len(src_model.layers) == len(dst_model.layers)

    copied = dict()
    layers = zip(src_model.layers, dst_model.layers)
    for src_layer, dst_layer in layers:
        if (len(src_layer.get_weights()) == 0):
            # no weight, skip
            continue

        dst_layer.set_weights(src_layer.get_weights())
        copied[src_layer.name] = dst_layer.name

    return copied


def int_to_onehot(seqs, dim=4):
    """One-hot encodes array of integer sequences.

    Takes array [nb_seq, seq_len] of integer sequence end encodes them one-hot.
    Special nucleotides (int > 4) will be encoded as [0, 0, 0, 0].

    Paramters
    ---------
    seqs: :class:`numpy.ndarray`
        [nb_seq, seq_len] :class:`numpy.ndarray` of integer sequences.
    dim: int
        Number of nucleotides

    Returns
    -------
    :class:`numpy.ndarray`
        [nb_seq, seq_len, dim] :class:`numpy.ndarray` of one-hot encoded
        sequences.
    """
    seqs = np.atleast_2d(np.asarray(seqs))
    n = seqs.shape[0]
    l = seqs.shape[1]
    enc_seqs = np.zeros((n, l, dim), dtype='int8')
    for i in range(dim):
        t = seqs == i
        enc_seqs[t, i] = 1
    return enc_seqs


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


def get_sample_weights(y, class_weights=None):
    """Compute sample weights for model training.

    Computes sample weights given  a vector of output labels `y`. Sets weights
    of samples without label (`CPG_NAN`) to zero.

    Parameters
    ----------
    y: :class:`numpy.ndarray`
        1d numpy array of output labels.
    class_weights: dict
        Weight of output classes, e.g. methylation states.

    Returns
    -------
    :class:`numpy.ndarray`
        Sample weights of size `y`.
    """
    y = y[:]
    sample_weights = np.ones(y.shape, dtype=K.floatx())
    sample_weights[y == CPG_NAN] = 0  # K.epsilon()
    if class_weights is not None:
        for cla, weight in class_weights.items():
            sample_weights[y == cla] = weight
    return sample_weights


def get_output_distribution(y):
    y = y[:]
    label_distribution = np.ones((len(y), 2), dtype=K.floatx())
    label_distribution[:, 0] = y
    label_distribution[:, 1] = 1 - y
    label_distribution[y == CPG_NAN] = CPG_NAN
    return label_distribution


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
        order = ['model.h5', 'model_weights.h5', 'model_weights_val.h5',
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


def get_objectives(output_names, loss_fn):
    """Return training objectives for a list of output names.

    Returns
    -------
    dict
        dict with `output_names` as keys and the name of the assigned Keras
        objective as values.
    """
    objectives = dict()
    for output_name in output_names:
        _output_name = output_name.split(OUTPUT_SEP)
        if _output_name[0] in ['bulk']:
            objective = 'mean_squared_error'
        elif _output_name[-1] in ['mean', 'var']:
            objective = 'mean_squared_error'
        elif _output_name[-1] in ['cat_var']:
            objective = 'categorical_crossentropy'
        elif loss_fn == 'kl_divergence':
            objective = 'kl_divergence'
        else:
            objective = 'binary_crossentropy'
        objectives[output_name] = objective
    return objectives


def add_output_layers(stem, output_names, loss_fn=None, init='glorot_uniform'):
    """Add and return outputs to a given layer.

    Adds output layer for each output in `output_names` to layer `stem`.

    Parameters
    ----------
    stem: Keras layer
        Keras layer to which output layers are added.
    output_names: list
        List of output names.
    loss_fn: objective function
        example: kl_divergence

    Returns
    -------
    list
        Output layers added to `stem`.
    """
    outputs = []
    for output_name in output_names:
        _output_name = output_name.split(OUTPUT_SEP)
        if _output_name[-1] in ['entropy']:
            x = kl.Dense(1, kernel_initializer=init, activation='relu')(stem)
        elif _output_name[-1] in ['var']:
            x = kl.Dense(1, kernel_initializer=init)(stem)
            x = ScaledSigmoid(0.251, name=output_name)(x)
        elif _output_name[-1] in ['cat_var']:
            x = kl.Dense(3, kernel_initializer=init,
                         activation='softmax',
                         name=output_name)(stem)
        elif loss_fn == 'kl_divergence':
            x = kl.Dense(2, kernel_initializer=init,
                         activation='softmax',
                         name=output_name)(stem)
        else:
            x = kl.Dense(1, kernel_initializer=init,
                         activation='sigmoid',
                         name=output_name)(stem)
        outputs.append(x)
    return outputs







def read_from(reader, nb_sample=None):
    """Read `nb_sample` samples from `reader`."""
    data = None
    nb_seen = 0
    for data_batch in reader:
        if not isinstance(data_batch, list):
            data_batch = list(data_batch)

        if not data:
            data = [dict() for i in range(len(data_batch))]
        for i in range(len(data_batch)):
            add_to_dict(data_batch[i], data[i])

        nb_seen += len(list(data_batch[0].values())[0])
        if nb_sample and nb_seen >= nb_sample:
            break

    for i in range(len(data)):
        data[i] = stack_dict(data[i])
        if nb_sample:
            for key, value in data[i].items():
                data[i][key] = value[:nb_sample]
    return data

def read(data_files, names, nb_sample=None, batch_size=1024, *args, **kwargs):
    data_reader = reader(data_files, names, batch_size=batch_size,
                         nb_sample=nb_sample, loop=False, *args, **kwargs)
    return read_from(data_reader, nb_sample)


def copy_weights(src_model, dst_model, must_exist=True):
    """Copy weights from `src_model` to `dst_model`.

    Parameters
    ----------
    src_model
        Keras source model.
    dst_model
        Keras destination model.
    must_exist: bool
        If `True`, raises `ValueError` if a layer in `dst_model` does not exist
        in `src_model`.

    Returns
    -------
    list
        Names of layers that were copied.
    """
    copied = []
    for dst_layer in dst_model.layers:
        for src_layer in src_model.layers:
            if src_layer.name == dst_layer.name:
                break
        if not src_layer:
            if must_exist:
                tmp = 'Layer "%s" not found!' % (src_layer.name)
                raise ValueError(tmp)
            else:
                continue
        dst_layer.set_weights(src_layer.get_weights())
        copied.append(dst_layer.name)
    return copied


def is_input_layer(layer):
    """Test if `layer` is an input layer."""
    return isinstance(layer, tf.keras.layers.InputLayer)


def is_output_layer(layer, model):
    """Test if `layer` is an output layer."""
    return layer.name in model.output_names


class Model(object):
    """Abstract model call.

    Abstract class of DNA, CpG, and Joint models.

    Parameters
    ----------
    dropout: float
        Dropout rate.
    l1_decay: float
        L1 weight decay.
    l2_decay: float
        L2 weight decay.
    init: str
        Name of Keras initialization.
    """

    def __init__(self, dropout=0.0, l1_decay=0.0, l2_decay=0.0,
                 batch_norm=False, init='glorot_uniform'):
        self.dropout = dropout
        self.l1_decay = l1_decay
        self.l2_decay = l2_decay
        self.batch_norm = batch_norm
        self.init = init
        self.name = self.__class__.__name__
        self.scope = None

    def inputs(self, *args, **kwargs):
        """Return list of Keras model inputs."""
        pass

    def _build(self, input, output):
        """Build final model at the end of `__call__`."""
        model = km.Model(input, output, name=self.name)
        if self.scope:
            for layer in model.layers:
                if not is_input_layer(layer):
                    layer._name = '%s/%s' % (self.scope, layer.name)
        return model

    def __call__(self, inputs=None):
        """Build model.

        Parameters
        ----------
        inputs: list
            Keras model inputs
        """
        pass


def encode_replicate_names(replicate_names):
    """Encode list of replicate names as single string.

    .. note:: Deprecated
        This function is used to support legacy models and will be removed in
        the future.
    """
    return '--'.join(replicate_names)


def decode_replicate_names(replicate_names):
    """Decode string of replicate names and return names as list.

    .. note:: Deprecated
        This function is used to support legacy models and will be removed in
        the future.
    """
    return replicate_names.split('--')


def hnames_to_names(hnames):
    """Flattens `dict` `hnames` of hierarchical names.

    Converts hierarchical `dict`, e.g. hnames={'a': ['a1', 'a2'], 'b'}, to flat
    list of keys for accessing HDF5 file, e.g. ['a/a1', 'a/a2', 'b']
    """
    names = []
    for key, value in six.iteritems(hnames):
        if isinstance(value, dict):
            for name in hnames_to_names(value):
                names.append('%s/%s' % (key, name))
        elif isinstance(value, list):
            for name in value:
                names.append('%s/%s' % (key, name))
        elif isinstance(value, str):
            names.append('%s/%s' % (key, value))
        else:
            names.append(key)
    return names


def reader(data_files, names, batch_size=128, nb_sample=None, shuffle=False,
           loop=False):
    if isinstance(names, dict):
        names = hnames_to_names(names)
    else:
        names = to_list(names)
    # Copy, since list will be changed if shuffle=True
    data_files = list(to_list(data_files))

    # Check if names exist
    h5_file = h5.File(data_files[0], 'r')
    for name in names:
        if name not in h5_file:
            raise ValueError('%s does not exist!' % name)
    h5_file.close()

    if nb_sample:
        # Select the first k files s.t. the total sample size is at least
        # nb_sample. Only these files will be shuffled.
        _data_files = []
        nb_seen = 0
        for data_file in data_files:
            h5_file = h5.File(data_file, 'r')
            nb_seen += len(h5_file[names[0]])
            h5_file.close()
            _data_files.append(data_file)
            if nb_seen >= nb_sample:
                break
        data_files = _data_files
    else:
        nb_sample = np.inf

    file_idx = 0
    nb_seen = 0
    while True:
        if shuffle and file_idx == 0:
            np.random.shuffle(data_files)

        h5_file = h5.File(data_files[file_idx], 'r')
        data_file = dict()
        for name in names:
            data_file[name] = h5_file[name]
        nb_sample_file = len(list(data_file.values())[0])

        if shuffle:
            # Shuffle data within the entire file, which requires reading
            # the entire file into memory
            idx = np.arange(nb_sample_file)
            np.random.shuffle(idx)
            for name, value in six.iteritems(data_file):
                data_file[name] = value[:len(idx)][idx]

        nb_batch = int(np.ceil(nb_sample_file / batch_size))
        for batch in range(nb_batch):
            batch_start = batch * batch_size
            nb_read = min(nb_sample - nb_seen, batch_size)
            batch_end = min(nb_sample_file, batch_start + nb_read)
            _batch_size = batch_end - batch_start
            if _batch_size == 0:
                break

            data_batch = dict()
            for name in names:
                data_batch[name] = data_file[name][batch_start:batch_end]
            yield data_batch

            nb_seen += _batch_size
            if nb_seen >= nb_sample:
                break

        h5_file.close()
        file_idx += 1
        assert nb_seen <= nb_sample
        if nb_sample == nb_seen or file_idx == len(data_files):
            if loop:
                file_idx = 0
                nb_seen = 0
            else:
                break


class DataReader(object):
    """Read data from `dcpg_data.py` output files.

    Generator to read data batches from `dcpg_data.py` output files. Reads data
    using :func:`hdf.reader` and pre-processes data.

    Parameters
    ----------
    output_names: list
        Names of outputs to be read.
    use_dna: bool
        If `True`, read DNA sequence windows.
    dna_wlen: int
        Maximum length of DNA sequence windows.
    replicate_names: list
        Name of cells (profiles) whose neighboring CpG sites are read.
    cpg_wlen: int
        Maximum number of neighboring CpG sites.
    cpg_max_dist: int
        Value to threshold the distance of neighboring CpG sites.
    encode_replicates: bool
        If `True`, encode replicated names in key of returned dict. This option
        is deprecated and will be removed in the future.

    Returns
    -------
    tuple
        `dict` (`inputs`, `outputs`, `weights`), where `inputs`, `outputs`,
        `weights` is a `dict` of model inputs, outputs, and output weights.
        `outputs` and `weights` are not returned if `output_names` is undefined.
    """

    def __init__(self, output_names=None,
                 use_dna=True, dna_wlen=None,
                 replicate_names=None, cpg_wlen=None, cpg_max_dist=25000,
                 encode_replicates=False,
                 loss_fn=None):
        self.output_names = to_list(output_names)
        self.use_dna = use_dna
        self.dna_wlen = dna_wlen
        self.replicate_names = to_list(replicate_names)
        self.cpg_wlen = cpg_wlen
        self.cpg_max_dist = cpg_max_dist
        self.encode_replicates = encode_replicates
        self.loss_fn = loss_fn

    def _prepro_dna(self, dna):
        """Preprocess DNA sequence windows.

        Slices DNA sequence window if `self.dna_wlen` is defined and one-hot
        encodes sequences.

        Parameters
        ----------
        dna: :class:`numpy.ndarray`
            :class:`numpy.ndarray` of size [nb_window, window_len] with integer
            sequences windows.

        Returns
        -------
        :class:`numpy.ndarray`
            :class:`numpy.ndarray` of size [nb_window, window_len, 4] with
            one-hot encoded sequences.
        """
        if self.dna_wlen:
            cur_wlen = dna.shape[1]
            center = cur_wlen // 2
            delta = self.dna_wlen // 2
            dna = dna[:, (center - delta):(center + delta + 1)]
        return int_to_onehot(dna)

    def _prepro_cpg(self, states, dists):
        """Preprocess the state and distance of neighboring CpG sites.

        Parameters
        ----------
        states: list
            List of CpG states of all replicates.
        dists: list
            List of CpG distances of all replicates.

        Returns
        -------
        prepro_states: list
            List of preprocessed CpG states of all replicates.
        prepro_dists: list
            List of preprocessed CpG distances of all replicates.
        """
        prepro_states = []
        prepro_dists = []
        for state, dist in zip(states, dists):
            nan = state == CPG_NAN
            if np.any(nan):
                # Set CpG neighbors at the flanks of a chromosome to 0.5
                state[nan] = 0.5
                dist[nan] = self.cpg_max_dist
            dist = np.minimum(dist, self.cpg_max_dist) / self.cpg_max_dist
            prepro_states.append(np.expand_dims(state, 1))
            prepro_dists.append(np.expand_dims(dist, 1))
        prepro_states = np.concatenate(prepro_states, axis=1)
        prepro_dists = np.concatenate(prepro_dists, axis=1)
        if self.cpg_wlen:
            center = prepro_states.shape[2] // 2
            delta = self.cpg_wlen // 2
            tmp = slice(center - delta, center + delta)
            prepro_states = prepro_states[:, :, tmp]
            prepro_dists = prepro_dists[:, :, tmp]
        return (prepro_states, prepro_dists)

    @threadsafe_generator
    def __call__(self, data_files, class_weights=None, *args, **kwargs):
        """Return generator for reading data from `data_files`.

        Parameters
        ----------
        data_files: list
            List of data files to be read.
        class_weights: dict
            dict of dict with class weights of individual outputs.
        *args: list
            Unnamed arguments passed to :func:`hdf.reader`
        *kwargs: dict
            Named arguments passed to :func:`hdf.reader`

        Returns
        -------
        generator
            Python generator for reading data.
        """
        names = []
        if self.use_dna:
            names.append('inputs/dna')

        if self.replicate_names:
            for name in self.replicate_names:
                names.append('inputs/cpg/%s/state' % name)
                names.append('inputs/cpg/%s/dist' % name)

        if self.output_names:
            for name in self.output_names:
                names.append('outputs/%s' % name)

        for data_raw in reader(data_files, names, *args, **kwargs):
            inputs = dict()

            if self.use_dna:
                inputs['dna'] = self._prepro_dna(data_raw['inputs/dna'])

            if self.replicate_names:
                states = []
                dists = []
                for name in self.replicate_names:
                    tmp = 'inputs/cpg/%s/' % name
                    states.append(data_raw[tmp + 'state'])
                    dists.append(data_raw[tmp + 'dist'])
                states, dists = self._prepro_cpg(states, dists)
                if self.encode_replicates:
                    # DEPRECATED: to support loading data for legacy models
                    tmp = '/' + encode_replicate_names(self.replicate_names)
                else:
                    tmp = ''
                inputs['cpg/state%s' % tmp] = states
                inputs['cpg/dist%s' % tmp] = dists

            if not self.output_names:
                yield inputs
            else:
                outputs = dict()
                weights = dict()

                for name in self.output_names:
                    outputs[name] = data_raw['outputs/%s' % name]
                    cweights = class_weights[name] if class_weights else None
                    weights[name] = get_sample_weights(outputs[name], cweights)
                    if self.loss_fn == 'kl_divergence':
                        outputs[name] = get_output_distribution(outputs[name])
                    if name.endswith('cat_var'):
                        output = outputs[name]
                        outputs[name] = to_categorical(output, 3)
                        outputs[name][output == CPG_NAN] = 0

                yield (inputs, outputs, weights)


def data_reader_from_model(model, outputs=True, replicate_names=None, loss_fn=None):
    """Return :class:`DataReader` from `model`.

    Builds a :class:`DataReader` for reading data for `model`.

    Parameters
    ----------
    model: :class:`Model`.
        :class:`Model`.
    outputs: bool
        If `True`, return output labels.
    replicate_names: list
        Name of input cells of `model`.
    loss_fn: str
        example: kl_divergence

    Returns
    -------
    :class:`DataReader`
        Instance of :class:`DataReader`.
    """
    use_dna = False
    dna_wlen = None
    cpg_wlen = None
    output_names = None
    encode_replicates = False

    input_shapes = to_list(model.input_shape)
    for input_name, input_shape in zip(model.input_names, input_shapes):
        if input_name == 'dna':
            # Read DNA sequences.
            use_dna = True
            dna_wlen = input_shape[1]
        elif input_name.startswith('cpg/state/'):
            # DEPRECATED: legacy model. Decode replicate names from input name.
            replicate_names = decode_replicate_names(
                input_name.replace('cpg/state/', ''))
            assert len(replicate_names) == input_shape[1]
            cpg_wlen = input_shape[2]
            encode_replicates = True
        elif input_name == 'cpg/state':
            # Read neighboring CpG sites.
            if not replicate_names:
                raise ValueError('Replicate names required!')
            if len(replicate_names) != input_shape[1]:
                tmp = '{r} replicates found but CpG model was trained with' \
                      ' {s} replicates. Use `--nb_replicate {s}` or ' \
                      ' `--replicate_names` option to select {s} replicates!'
                tmp = tmp.format(r=len(replicate_names), s=input_shape[1])
                raise ValueError(tmp)
            cpg_wlen = input_shape[2]

    if outputs:
        # Return output labels.
        output_names = model.output_names

    # loss_fn = model.loss[model.output_names[0]]

    return DataReader(output_names=output_names,
                      use_dna=use_dna,
                      dna_wlen=dna_wlen,
                      cpg_wlen=cpg_wlen,
                      replicate_names=replicate_names,
                      encode_replicates=encode_replicates,
                      loss_fn=loss_fn)
