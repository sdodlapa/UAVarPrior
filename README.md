# UAVarPrior: Uncertainty-Aware Variational Prior

UAVarPrior is a framework for incorporating uncertainty-aware variational priors into machine learning models. This approach provides robust uncertainty quantification while maintaining predictive performance.

## Features

- **Variational Prior Integration**: Leverage probabilistic modeling with variational priors
- **Uncertainty Quantification**: Capture and quantify both aleatoric and epistemic uncertainty
- **Flexible Configuration**: YAML-based configuration for reproducible experiments
- **Command-line Interface**: Intuitive CLI for training, evaluation, and inference

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/UAVarPrior.git
cd UAVarPrior

# Install the package
pip install -e .
```

## Large Files Handling

This repository uses `.gitignore` to exclude large files that exceed GitHub's 100MB limit. Key large files are stored in designated `outputs` directories that are not tracked by Git.

### Locations for Large Files

- **Data Files**: Large data files should be stored in `uavarprior/data/outputs/`
- **Model Weights**: Large model files should be stored in `uavarprior/model/outputs/`
- **Analysis Results**: Large analysis files (like pickle files) should be stored in `uavarprior/interpret/outputs/`

### Sharing Large Files

For collaboration, large files can be shared via:
1. Cloud storage (Google Drive, Dropbox, etc.)
2. Institutional file sharing services
3. Data repositories like Zenodo or Figshare

### Using Git LFS (Optional)

For advanced users, Git Large File Storage (LFS) can be used to track large files:

```bash
# Install Git LFS
git lfs install

# Track large file types
git lfs track "*.pkl"
git lfs track "*.parquet.gz"

# Make sure to commit the .gitattributes file
git add .gitattributes
```
## Quick Start
Basic Usage

```bash
# Run with configuration file
uavarprior run config/default.yaml

# Override configuration parameters
uavarprior run config/default.yaml -o training.learning_rate=0.001 -o training.epochs=100

# Validate configuration file
uavarprior validate config/my_experiment.yaml
```

## Example Configuration

```bash
model:
  name: "variational_model"
  layers: [256, 128, 64]
  dropout_rate: 0.1
  
training:
  batch_size: 64
  learning_rate: 0.001
  epochs: 50
  optimizer: "adam"
  
data:
  dataset_path: "/path/to/dataset"
  validation_split: 0.2
  ```