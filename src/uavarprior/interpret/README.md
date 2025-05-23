# Large Analysis Files

This directory may contain large data and analysis files that exceed GitHub's file size limit (100MB).

## Large File Handling Strategy

For files larger than 100MB, we use the following approach:

1. The files are excluded from git tracking in `.gitignore`
2. Large output files should be stored in one of these ways:
   - On a shared storage accessible to the team
   - On a cloud storage service (Google Drive, Dropbox, etc.)
   - Generated using code provided in this repository

## Current Large Files

- `unique_names_group_1.pkl` (144.02 MB): Set of unique variant names across multiple files
- `variant_membership_matrix_group_1.pkl` (841.28 MB): Binary membership matrix of variants across files

These files can be generated using the `signVar_by_profile.ipynb` notebook in this directory.

## Generating the Files

To regenerate the large files locally, run:

```bash
cd uavarprior/interpret
jupyter notebook signVar_by_profile.ipynb
```

And execute all cells to generate the analysis files.
