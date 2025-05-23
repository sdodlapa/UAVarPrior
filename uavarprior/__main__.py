#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main entry point for UAVarPrior when run as a module.

This allows running the package as:
    python -m uavarprior
"""

from uavarprior.cli import cli

if __name__ == "__main__":
    cli()