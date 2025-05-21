# Large Model Files Management

This document provides instructions for handling large model files in the UAVarPrior project.

## Output Directory

Large model files should be stored in the `uavarprior/model/outputs/` directory, which is not tracked by Git.

## Contents

The outputs directory should store:
- Model weights (`.pt`, `.pth`, `.ckpt`)
- Trained models
- Model checkpoints
- Large model configuration files

## Guidelines

1. Files in the outputs directory are **not tracked** by Git (as specified in the root `.gitignore` file)
2. Files larger than 100MB **cannot** be pushed to GitHub
3. Share these files with collaborators via cloud storage or other file sharing mechanisms
4. Consider using relative paths in your code to reference these files:

```python
import os
import torch

# Get the absolute path to the outputs directory
model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../outputs')

# Example: Save a model
model_path = os.path.join(model_dir, 'my_trained_model.pt')
torch.save(model.state_dict(), model_path)

# Example: Load a model
model.load_state_dict(torch.load(model_path))
```

## Model Versioning

When saving models, consider using a versioning scheme:
- Include the date: `model_20250520.pt`
- Include the version: `model_v1.2.pt`
- Include hyperparameters: `model_lr0.001_epochs100.pt`
