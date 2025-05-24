"""
Utility functions for data module.
"""

def formatChrom(chrom):
    """
    Format chromosome name to ensure consistency.
    
    Parameters
    ----------
    chrom : str
        Chromosome name.
        
    Returns
    -------
    str
        Formatted chromosome name.
    """
    if isinstance(chrom, int):
        chrom = str(chrom)
    
    if not chrom.startswith('chr'):
        chrom = 'chr' + chrom
        
    return chrom
