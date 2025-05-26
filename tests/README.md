# UAVarPrior Test Structure

This directory contains all tests for the UAVarPrior project, organized by test type and functionality.

## Directory Structure

```
tests/
├── conftest.py                 # Pytest configuration and fixtures
├── unit/                       # Unit tests for individual modules
│   ├── data/                   # Data handling tests
│   ├── interpret/              # Interpretation module tests
│   ├── predict/                # Prediction module tests
│   └── *.py                    # Core functionality unit tests
├── integration/                # Integration tests
│   └── *.py                    # Cross-module integration tests
├── functional/                 # End-to-end functional tests
│   └── *.py                    # Complete workflow tests
├── legacy/                     # Legacy/deprecated tests
│   └── *.py                    # Old tests kept for reference
├── smoke/                      # Quick smoke tests
│   └── *.py                    # Basic functionality verification
└── fixtures/                   # Test data and fixtures
    └── ...                     # Test data files
```

## Running Tests

### All Tests
```bash
pytest tests/
```

### Specific Test Categories
```bash
# Unit tests only
pytest tests/unit/

# Integration tests only
pytest tests/integration/

# Functional tests only
pytest tests/functional/

# Smoke tests only
pytest tests/smoke/
```

### Individual Test Files
```bash
pytest tests/unit/test_import.py -v
```

## Test Markers

Tests are marked with the following categories:
- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.functional` - Functional tests
- `@pytest.mark.slow` - Tests that take longer to run

### Running by Marker
```bash
# Skip slow tests
pytest -m "not slow"

# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration
```

## Test Fixtures

Common fixtures are defined in `conftest.py`:
- `temp_dir` - Temporary directory for test files
- `sample_config_dir` - Path to sample configuration files
- `test_data_dir` - Path to test data fixtures
- `mock_model_path` - Mock model file for testing

## Adding New Tests

1. **Unit Tests**: Add to `tests/unit/` - test individual functions/classes
2. **Integration Tests**: Add to `tests/integration/` - test module interactions
3. **Functional Tests**: Add to `tests/functional/` - test complete workflows
4. **Test Data**: Add to `tests/fixtures/` - test data files

## Test Naming Convention

- Test files: `test_<functionality>.py`
- Test functions: `test_<specific_behavior>()`
- Test classes: `Test<ClassName>`
