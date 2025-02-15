'''
Utils for handling data

Created on May 25, 2021

@author: jsun
'''

from itertools import compress
import re
import numpy as np
import pandas as pd

def formatChrom(chrom):
    """Format chromosome name to follow the name convention used
    in Essemble data.

    Makes name upper case, e.g. 'mt' -> 'MT' and removes 'chr',
    e.g. 'chr1' -> '1'.
    """
    if isinstance(chrom, str):
        return re.sub('^CHR', '', chrom) #chrom.str.upper())
    elif isinstance(chrom, pd.Series):
        chrom = chrom.astype(str)
        chrom = chrom.str.upper()
        return chrom.replace(r'CHR', '', regex=True) # Sanjeev 08/19/2022

def isBedgraph(filename):
    """Test if `filename` is a bedGraph file.

    bedGraph files are assumed to start with 'track type=bedGraph'

    Note
    -------
    DeepCpG function is_bedgraph
    """
    if isinstance(filename, str):
        with open(filename) as f:
            line = f.readline()
    else:
        pos = filename.tell()
        line = filename.readline()
        if isinstance(line, bytes):
            line = line.decode()
        filename.seek(pos)
    return re.match(r'track\s+type=bedGraph', line) is not None


def isBinary(values):
    """Check if values in array `values` are binary, i.e. zero or one."""
    return np.any((values > 0) & (values < 1))


def isInt(value):
    try:
        int(value)
        return True
    except ValueError:
        return False


def sampleByChrom(df, nSamp):
    """Randomly sample `nSamp` samples from each chromosome.

    Samples `nSamp` records from :class:`pandas.DataFrame` which must
    contain a column with name 'chrom'.

    Note
    -------
    Function sampleByChrom in DeepCpG
    """

    def sampleFromDf(df):
        if len(df) <= nSamp:
            return df
        idx = np.random.choice(len(df), nSamp, replace=False)
        return df.iloc[idx]

    outDf = df.groupby('chrom', as_index=False).apply(sampleFromDf)
    outDf.index = range(len(outDf))
    return outDf


def sortChrom(chroms):
    '''
    Sort chromosomes in the order of chr1, chr2, chr3, ...
    '''
    startWithChr = False
    if chroms[0].startswith('chr'):
        # strip chr
        chroms = [chrom[3:] for chrom in chroms]
        startWithChr = True
    isNumChr = [isInt(chrom) for chrom in chroms]
    numChr = list(compress(chroms, isNumChr))
    letChr = list(compress(chroms, [not elem for elem in isNumChr]))
    numChr = [int(chrom) for chrom in numChr]
    numChr.sort()
    letChr.sort()
    numChr = [str(chrom) for chrom in numChr]
    chroms = numChr + letChr

    if not startWithChr:
        return chroms

    chroms = ['chr' + chrom for chrom in chroms]

    return chroms