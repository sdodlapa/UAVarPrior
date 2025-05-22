"""Command line interface for UAVarPrior.

Why does this file exist, and why not put this in ``__main__``? You might be tempted to import things from ``__main__``
later, but that will cause problems--the code will get executed twice:

- When you run ``python3 -m uavarprior`` python will execute``__main__.py`` as a script. That means there won't be any
  ``uavarprior.__main__`` in ``sys.modules``.
- When you import __main__ it will get executed again (as a module) because
  there's no ``uavarprior.__main__`` in ``sys.modules``.

.. seealso:: http://click.pocoo.org/5/setuptools/#setuptools-integration
"""

import click

from uavarprior import __version__
from uavarprior.setup import load_path, parse_configs_and_run


@click.command()
@click.version_option(__version__)
@click.argument('path', type=click.Path(exists=True, file_okay=True, dir_okay=False))
# @click.option('--lr', type=float, help='If training, the optimizer learning rate', show_default=True)
def main(path):
    """Build the model and trains it using user-specified input data."""
    configs = load_path(path, instantiate=False)
    parse_configs_and_run(configs)


if __name__ == "__main__":
    main()
