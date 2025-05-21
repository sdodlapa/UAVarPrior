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

## Project Structure

The project is organized into several modules:

- **data**: For data processing and management
- **model**: For model definitions and training
- **interpret**: For result analysis and interpretation

Each module has a `docs` directory with documentation and an `outputs` directory for large files.

## Large File Management

We have implemented a comprehensive system for managing large files:

1. **Output Directories**: Each module has an `outputs` directory for storing large files
2. **Documentation**: See `uavarprior/*/docs/LARGE_FILES.md` for guidelines
3. **Utility Script**: Use `scripts/manage_large_files.py` to find and move large files

Example usage:
```bash
# Find large files
python scripts/manage_large_files.py find --threshold 50

# Move large files to appropriate output directories
python scripts/manage_large_files.py move --threshold 50 --dry-run
python scripts/manage_large_files.py move --threshold 50
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