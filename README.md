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