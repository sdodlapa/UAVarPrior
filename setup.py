import os

from Cython.Distutils import build_ext
import numpy as np
from setuptools import Extension
from setuptools import find_packages
from setuptools import setup

with open(os.path.join(os.path.dirname(__file__), "README.md"),
          encoding='utf-8') as readme:
    long_description = readme.read()

genome_module = Extension(
    "uavarprior.data.sequences._sequence",
    ["src/uavarprior/data/sequences/_sequence.pyx"],
    include_dirs=[np.get_include()])

genomic_features_module = Extension(
    "uavarprior.data.targets._genomic_features",
    ["src/uavarprior/data/targets/_genomic_features.pyx"],
    include_dirs=[np.get_include()])

ext_modules = [genome_module, genomic_features_module]
cmdclass = {'build_ext': build_ext}

setup(name="uavarprior",
      version="0.2.0",
      long_description=long_description,
      long_description_content_type='text/markdown',
      description=("Uncertainty-Aware Variational Prior framework for "
                   "deep learning sequence models"),
      packages=find_packages(exclude=['src']),
      url="https://github.com/FunctionLab/UAVarPrior",
      package_data={
          "uavarprior.interpret": [
              "data/gencode_v28_hg38/*",
              "data/gencode_v28_hg19/*"
          ],
          "uavarprior.sequences": [
              "data/*"
          ]
      },
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: BSD License",
          "Topic :: Scientific/Engineering :: Bio-Informatics"
      ],
      ext_modules=ext_modules,
      cmdclass=cmdclass,
      install_requires=[
          "cython>=0.27.3",
          'click>=8.0.0',
          "h5py>=3.1.0",
          "matplotlib>=2.2.3",
          "numpy>=1.19.0",
          "pandas>=1.3.0",
          "torchinfo",
          "plotly",
          "pyfaidx",
          "pytabix",
          "pyyaml>=5.1",
          "scikit-learn>=1.0.0",
          "scipy>=1.7.0",
          "seaborn",
          "statsmodels",
          "torch>=1.10.0",
          "bioPython>=1.73",
          "pydantic>=1.9.0",
      ],
      entry_points={
          'console_scripts': [
              'uavarprior=uavarprior.cli:cli',
          ],
      },
      )
