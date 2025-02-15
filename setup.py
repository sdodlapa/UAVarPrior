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
    "fugep.data.sequences._sequence",
    ["fugep/data/sequences/_sequence.pyx"],
    include_dirs=[np.get_include()])

genomic_features_module = Extension(
    "fugep.data.targets._genomic_features",
    ["fugep/data/targets/_genomic_features.pyx"],
    include_dirs=[np.get_include()])

ext_modules = [genome_module, genomic_features_module]
cmdclass = {'build_ext': build_ext}

setup(name="fugep",
      version="0.4.8",
      long_description=long_description,
      long_description_content_type='text/markdown',
      description=("framework for developing sequence-level "
                   "deep learning networks"),
      packages=find_packages(),
      url="https://github.com/FunctionLab/selene",
      package_data={
          "fugep.interpret": [
              "data/gencode_v28_hg38/*",
              "data/gencode_v28_hg19/*"
          ],
          "fugep.sequences": [
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
          'click',
          "h5py",
          "matplotlib>=2.2.3",
          "numpy",
          "pandas",
          "torchinfo",
          "plotly",
          "pyfaidx",
          "pytabix",
          "pyyaml>=5.1",
          "scikit-learn",
          "scipy",
          "seaborn",
          "statsmodels",
          "torch>=0.4.1",
          "bioPython >= 1.73",
      ],
      entry_points={
          'console_scripts': [
              'fugep = fugep.cli:main',
          ],
      },
      )
