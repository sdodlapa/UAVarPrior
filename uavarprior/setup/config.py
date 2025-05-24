#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration utilities for UAVarPrior.

This module provides utilities for loading and processing configurations.
"""
import os
import re
import warnings
import yaml
import six
from collections import namedtuple
from typing import Dict, Any, Optional, Union

SCIENTIFIC_NOTATION_REGEXP = r"^[\-\+]?(\d+\.?\d*|\d*\.?\d+)?[eE][\-\+]?\d+$"
IS_INITIALIZED = False

_BaseProxy = namedtuple("_BaseProxy", ["callable", "positionals", "keywords",
                                     "yaml_src"])

class _Proxy(_BaseProxy):
    """
    Helper class for object instantiation from YAML.
    """
    __slots__ = ()

    def __new__(cls, callable=None, positionals=None, keywords=None, yaml_src=None):
        if positionals is None:
            positionals = ()
        if keywords is None:
            keywords = {}
        return super(_Proxy, cls).__new__(cls, callable, positionals, keywords, yaml_src)


class _Proxy(_BaseProxy):
    """An intermediate representation between initial YAML parse and
    object instantiation.

    Parameters
    ----------
    callable : callable
        The function/class to call to instantiate this node.
    positionals : iterable
        Placeholder for future support for positional
        arguments (`*args`).
    keywords : dict-like
        A mapping from keywords to arguments (`**kwargs`), which may be
        `_Proxy`s or `_Proxy`s nested inside `dict` or `list` instances.
        Keys must be strings that are valid Python variable names.
    yaml_src : str
        The YAML source that created this node, if available.
    """
    __slots__ = []

    def __hash__(self):
        return hash((self.callable, id(self)))


class _RecursionGuard(object):
    """A utility class to handle recursive objects."""

    def __init__(self, bindings=None):
        if bindings is None:
            bindings = {}
        self.bindings = bindings

    def __call__(self, proxy):
        if proxy in self.bindings:
            return self.bindings[proxy]
        return None


def _instantiate_proxy_tuple(proxy, bindings=None):
    """
    Instantiate a proxy tuple by calling its callable with appropriate
    parameters.

    Parameters
    ----------
    proxy : _Proxy object
        A proxy object that was created by the yaml parser.
    bindings : dict, optional
        A dictionary mapping previously instantiated proxy objects to
        their instantiated values.

    Returns
    -------
    object
        The result of calling the callable on the arguments.
    """
    if bindings is None:
        bindings = {}
    if proxy in bindings:
        return bindings[proxy]
    recursion_guard = _RecursionGuard(bindings)
    callable_f = proxy.callable
    positionals = [instantiate(value, bindings)
                   for value in proxy.positionals]
    keywords = {k: instantiate(v, bindings) for k, v in
                six.iteritems(proxy.keywords)}
    try:
        obj = callable_f(*positionals, **keywords)
    except TypeError as e:
        # Reraise the error with a more informative traceback.
        message = str(e) + "\nTrying to instantiate " + str(proxy.callable)
        if proxy.yaml_src is not None:
            message += " from source:\n" + proxy.yaml_src
        raise TypeError(message)
    bindings[proxy] = obj
    return obj


def load_path(path: str, instantiate_obj: bool = False, **kwargs) -> Dict[str, Any]:
    """
    Load YAML configuration from path
    
    Args:
        path: Path to YAML configuration file
        instantiate_obj: Whether to instantiate objects in the configuration
        
    Returns:
        Configuration dictionary
    """
    # For backwards compatibility, allow 'instantiate' parameter
    if 'instantiate' in kwargs:
        instantiate_obj = kwargs['instantiate']
        
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    
    if instantiate_obj:
        config = instantiate(config)
    
    return config


def instantiate(proxy, bindings=None):
    """Instantiate a hierarchy of proxy objects.

    Parameters
    ----------
    proxy : object
        A `_Proxy` object or list/dict/literal. Strings are run through
        `_preprocess`.
    bindings : dict, optional
        A dictionary mapping previously instantiated `_Proxy` objects
        to their instantiated values.

    Returns
    -------
    obj : object
        The result object from recursively instantiating the object DAG.
    """
    if bindings is None:
        bindings = {}
    if isinstance(proxy, _Proxy):
        return _instantiate_proxy_tuple(proxy, bindings)
    elif isinstance(proxy, dict):
        # Recurse on the keys too, for backward compatibility.
        # Is the key instantiation feature ever actually used, by anyone?
        return dict((instantiate(k, bindings), instantiate(v, bindings))
                    for k, v in six.iteritems(proxy))
    elif isinstance(proxy, list):
        return [instantiate(v, bindings) for v in proxy]
    # In the future it might be good to consider a dict argument that provides
    # a type->callable mapping for arbitrary transformations like this.
    return proxy
