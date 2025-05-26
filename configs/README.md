# Configuration Management

This directory contains all configuration files for UAVarPrior, organized by purpose and usage type.

## Directory Structure

```
configs/
├── examples/           # Example configurations for new users
├── environments/       # Environment-specific configurations (GPU, CPU, etc.)
├── testing/           # Test-specific configurations
├── templates/         # Base templates for creating new configs
└── README.md          # This file
```

## Usage Guidelines

### Examples (`examples/`)
- **Purpose**: Ready-to-use example configurations for common tasks
- **Audience**: New users learning the system
- **Usage**: Copy and modify for your specific needs
- **Files**: 
  - `analyze_config.yml` - Basic analysis configuration
  - `train.yml` - Training configuration example
  - `simple_train_example.yml` - Minimal training setup
  - `variant_effect_prediction.yml` - Variant analysis example

### Environments (`environments/`)
- **Purpose**: Hardware/platform-specific configurations
- **Audience**: Users with specific computational environments
- **Usage**: Select based on your available hardware
- **Files**:
  - `selene-cpu.yml` - CPU-only configuration
  - `selene-gpu.yml` - GPU-enabled configuration
  - `methyl-*.yml` - Methylation-specific configurations
  - `peak-*.yml` - Peak analysis configurations

### Testing (`testing/`)
- **Purpose**: Configurations used by automated tests and CI/CD
- **Audience**: Developers and CI systems
- **Usage**: Referenced by test suites and development workflows
- **Files**: Various test configurations for different scenarios

### Templates (`templates/`)
- **Purpose**: Base templates for creating new configurations
- **Audience**: Advanced users and developers
- **Usage**: Starting point for custom configurations
- **Files**: Will contain minimal, well-commented template files

## Best Practices

1. **Use Examples First**: Start with configurations from `examples/` 
2. **Environment Selection**: Choose appropriate files from `environments/` based on your hardware
3. **Custom Configs**: Create custom configurations by copying from `examples/` or `templates/`
4. **Validation**: Use the CLI validation features to check your configurations
5. **Documentation**: Keep configurations well-commented for maintainability

## Configuration Validation

You can validate any configuration file using:

```bash
uavarprior validate-config path/to/your/config.yml
```

## Creating New Configurations

1. Start with a template from `templates/` or copy from `examples/`
2. Modify parameters for your specific use case
3. Validate the configuration before use
4. Test with a small dataset first

## Migration from Legacy Structure

This new structure consolidates the previous `config_examples/` and `configs/` directories:

- Old `config_examples/` → `configs/examples/`
- Test-related configs → `configs/testing/`
- Environment configs → `configs/environments/`
- New templates directory → `configs/templates/`

All existing configurations have been preserved and relocated appropriately.
