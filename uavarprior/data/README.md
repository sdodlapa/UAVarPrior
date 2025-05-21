--- a/uavarprior/data/README.md
+++ b/uavarprior/data/README.md
@@ -0,0 +1,30 @@
+# Large Data Files
+
+This directory may contain large data files that exceed GitHub's file size limit (100MB).
+
+## Large File Handling Strategy
+
+For files larger than 100MB, we use the following approach:
+
+1. The files are excluded from git tracking in `.gitignore`
+2. Large datasets should be stored in one of these ways:
+   - On a shared storage accessible to the team
+   - On a cloud storage service (Google Drive, Dropbox, etc.)
+   - Generated using code provided in this repository
+
+## Current Large Files
+
+- `combined_variant_positions.parquet.gz` (162.68 MB): Combined variant positions from ClinVar, COSMIC, and 1000 Genomes 
+  - This file can be generated using the `preprocess_input_var.ipynb` notebook
+  - Alternatively, download from: [shared drive location]
+
+## Generating the Files
+
+To regenerate the large files locally, run:
+
+```bash
+cd uavarprior/data
+jupyter notebook preprocess_input_var.ipynb
+```
+
+And execute all cells to generate the combined dataset.
