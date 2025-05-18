import os
import re
import numpy as np
import pandas as pd
import gzip
import time
import matplotlib.pyplot as plt



def regex_match(start_with, end_with, folder):
    m = re.search(start_with+end_with, folder)
    if m!=None:
        return True
    else:
        return False

_maf_re = re.compile(r'\bMAF=([^;]+)')

def extract_maf_regex(info_str):
    """
    Use a regular expression to grab the MAF= value.
    """
    m = _maf_re.search(info_str)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None

def classify_maf(maf):
    if maf < 0.001:
        return 'rare'
    elif maf > 0.05:
        return 'common'
    else:
        return 'none'
    




def read_vcf(vcf_file_path):
    """
    Read a gzipped VCF into a pandas DataFrame, dropping '##' meta‑lines.
    """
    with gzip.open(vcf_file_path, 'rt') as f:
        # keep only the header line (#CHROM ...) + data lines
        lines = [l for l in f if not l.startswith('##')]
    # join and read with pandas
    vcf_str = "".join(lines)
    df = pd.read_csv(io.StringIO(vcf_str), sep='\t')
    return df

def read_vcf(vcf_file_path):
    """
    Reads data from vcf file and parse lines into pandas dataframe
        Parameters:
            vcf_file_path: str
        
        Returns:
            pandas dataframe with columns ["CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO"]
    """
    VCF_REQUIRED_COLS = ["#CHROM", "POS", "ID", "REF", "ALT"]
    with gzip.open(vcf_file_path, 'rb') as file_handle:
        lines = file_handle.readlines()
        
        # handling first few lines
        index = 0
        for index, line in enumerate(lines):
            line = line.decode('utf8')
            if '#' not in line:
                break
            if "#CHROM" in line:
                cols = line.strip().split('\t')
                if cols[:5] != VCF_REQUIRED_COLS:
                    raise ValueError(
                        "First 5 columns in file {0} were {1}. "
                        "Expected columns: {2}".format(
                            input_path, cols[:5], VCF_REQUIRED_COLS))
                index += 1
                break  
         
    # handling remaining lines of vcf file
    variants = []
    for line in lines[index:]:
        line = line.decode('utf8')
        cols = line.strip().split('\t')
        variants.append(cols)

    df = pd.DataFrame(variants)
    print(df.shape)
    df.columns = ["CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO"]
    return df

def remove_extra_chroms(input_df):
    """
    Removes chromosomes other than 1-22
        Parameters:
            df: pandas dataframe with chrom as one of the columns
        Returns:
            Pandas dataframe
    """
    chroms = [str(chrom) for chrom in range(1, 23)]
    df = pd.DataFrame()
    input_df['chrom'] = input_df['chrom'].astype(str)
    for chrom in chroms:
        chr_df = input_df[input_df['chrom']==str(chrom)]
#         print(chrom, chr_df.shape)
        df = pd.concat([df, chr_df])
    df.reset_index(drop=True, inplace=True)
    for col in df.columns:
        try:
            if col.lower() == 'chrom':
                df['chrom'] = df['chrom'].astype(np.uint8)
            elif col.lower() in ['pos', 'start', 'end']:
                df[col] = df[col].astype(np.uint32)
        except:
            pass
    print('Extra chromosomes have been deleted and sorted by chrom and pos/start')
    for col in df.columns:
        try:
            if col.lower() in ['pos', 'start']:
                df.sort_values(by=['chrom', col], inplace=True, ignore_index=True)
        except:
            pass
    print('Data is sorted by chrom followed by pos/start')
    return df



release = 108
inPath = f'/scratch/ml-csm/datasets/genomics/ref-genome/human/GRCh38/ensembl/variants/release-{release}/chrom/'
files = os.listdir(inPath)
files.sort()
print(len(files))

chroms = [i for i in range(1, 23)]

num_var = {}
for chrom in chroms:
    file = f'homo_sapiens-chr{chrom}.vcf.gz'
    df = read_vcf(inPath + file)
    print(file, df.shape)
    num_var[chrom] = df.shape[0]


df_num_var = pd.DataFrame.from_dict(num_var, orient='index')
df_num_var.columns = ['num_var']
print(f"Total {df_num_var['num_var'].sum()}")
df_num_var.insert(loc=0, column='chrom', value=df_num_var.index)
df_num_var['chrom'] = df_num_var['chrom'].astype(str)
df_num_var.loc[len(df_num_var)] = ['Total', df_num_var['num_var'].sum()]
df_num_var.to_csv(f'GVE_peak/genomics/df_num_var_release{release}.csv', index=False)

df_num_var = pd.read_csv(f'GVE_peak/genomics/df_num_var_release{release}.csv')
df_num_var['var%'] = df_num_var['num_var']*100/df_num_var['num_var'].sum()
df_num_var, df_num_var['num_var'].sum(), df_num_var['var%'].sum()



inPath = '/scratch/ml-csm/projects/fgenom/gve/output/kmeans/annotations/ref_win/ref_anno_merged_var_count/'
files = os.listdir(inPath)

anno_dict = {}
for file in files:
    data = pd.read_csv(inPath+file, sep='\t')
    num_anno_var = data.iloc[:, -1].sum()
    print(file, num_anno_var)
    anno = file.split('.')[0]
    anno_dict[anno] = num_anno_var

df_anno_var = pd.DataFrame.from_dict(anno_dict, orient='index', columns=['num_var'])
df_anno_var.insert(loc=0, column='anno', value=df_anno_var.index)
df_anno_var.reset_index(drop=True, inplace=True)
df_anno_var['var%'] = df_anno_var['num_var']*100/df_num_var['num_var'].sum()


inPath = '/scratch/ml-csm/datasets/genomics/ref-genome/human/GRCh38/ensembl/variants/processed/'
df_maf = pd.read_parquet(inPath+'1000GENOMES-release114-maf.parquet.gz')
df_maf.dropna(inplace=True, ignore_index=True)
df_maf['category'] = df_maf['maf'].apply(classify_maf)

# Count rare, common, none, and total variants
rare_count = (df_maf['category'] == 'rare').sum()
common_count = (df_maf['category'] == 'common').sum()
none_count = (df_maf['category'] == 'none').sum()
total_count = len(df_maf)

# Print the results
print(f"Total: {total_count}, Rare: {rare_count}, Common: {common_count}, None: {none_count}")


maf_dict = {}
maf_dict['rare'] = rare_count
maf_dict['common'] = common_count
df_maf_var = pd.DataFrame.from_dict(maf_dict, orient='index', columns=['num_var'])
df_maf_var.insert(loc=0, column='anno', value=df_maf_var.index)
df_maf_var.reset_index(drop=True, inplace=True)
df_maf_var['var%'] = df_maf_var['num_var']*100/df_num_var['num_var'].sum()



# compute width ratios
w1 = len(df_num_var)
w2 = len(df_anno_var)
w3 = len(df_maf_var)
widths = [w1, w2, w3]

# scale factor for figure width per tick (tweak as needed)
scale = 0.3
fig_width = (w1 + w2 + w3) * scale

fig, axes = plt.subplots(
    1, 3,
    figsize=(fig_width, 5),
    sharey=False,
    gridspec_kw={'width_ratios': widths}
)

# 1) SNV distribution per chromosome
axes[0].bar(df_num_var['chrom'], df_num_var['var%'])
axes[0].set_xlabel('Chromosome')
axes[0].set_ylabel('% of SNVs')     # only on the leftmost plot

# 2) Variants in regulatory regions
axes[1].bar(df_anno_var['anno'], df_anno_var['var%'])
axes[1].set_xlabel('')
axes[1].tick_params(axis='x', rotation=45, labelsize=8)

# 3) Rare vs common variants
axes[2].bar(df_maf_var['anno'], df_maf_var['var%'])
axes[2].set_xlabel('')
axes[2].tick_params(axis='x', rotation=45, labelsize=8)

plt.tight_layout()
plt.savefig(f'GVE_peak/figures/SNV_distribution.pdf', bbox_inches='tight')