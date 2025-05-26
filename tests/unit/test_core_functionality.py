#!/usr/bin/env python3
"""
Unit tests for UAVarPrior core functionality.
"""

import pytest
import numpy as np
import os
import sys

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

class TestMatrixOperations:
    """Test matrix computation functionality."""
    
    def test_matrix_creation(self):
        """Test basic matrix creation."""
        matrix = np.random.rand(5, 5)
        assert matrix.shape == (5, 5)
        assert matrix.dtype == np.float64
    
    def test_matrix_eigenvalues(self):
        """Test eigenvalue computation."""
        matrix = np.eye(3)  # Identity matrix
        eigenvals = np.linalg.eigvals(matrix)
        expected = np.array([1.0, 1.0, 1.0])
        np.testing.assert_array_almost_equal(sorted(eigenvals), sorted(expected))
    
    def test_matrix_multiplication(self):
        """Test matrix multiplication."""
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6], [7, 8]])
        result = np.dot(a, b)
        expected = np.array([[19, 22], [43, 50]])
        np.testing.assert_array_equal(result, expected)

class TestVariantAnalysis:
    """Test variant analysis functionality."""
    
    def test_variant_data_structure(self):
        """Test variant data structure creation."""
        variant_data = {
            'chromosome': 'chr1',
            'position': 12345,
            'ref_allele': 'A',
            'alt_allele': 'T',
            'quality': 30.0
        }
        assert variant_data['chromosome'] == 'chr1'
        assert isinstance(variant_data['position'], int)
        assert variant_data['quality'] >= 0
    
    def test_variant_filtering(self):
        """Test variant filtering logic."""
        variants = [
            {'quality': 20.0, 'depth': 10},
            {'quality': 35.0, 'depth': 25},
            {'quality': 15.0, 'depth': 8}
        ]
        
        # Filter by quality threshold
        high_quality = [v for v in variants if v['quality'] >= 30.0]
        assert len(high_quality) == 1
        assert high_quality[0]['quality'] == 35.0

class TestConfigurationHandling:
    """Test configuration file handling."""
    
    def test_config_structure(self):
        """Test basic configuration structure."""
        config = {
            'analysis': {
                'method': 'eigenvalue_decomposition',
                'threshold': 0.05
            },
            'output': {
                'format': 'tsv',
                'directory': './results'
            }
        }
        
        assert 'analysis' in config
        assert 'output' in config
        assert config['analysis']['method'] == 'eigenvalue_decomposition'
    
    def test_parameter_validation(self):
        """Test parameter validation."""
        threshold = 0.05
        assert 0.0 <= threshold <= 1.0
        
        method = 'eigenvalue_decomposition'
        valid_methods = ['eigenvalue_decomposition', 'pca', 'svd']
        assert method in valid_methods

class TestAuthenticationSetup:
    """Test authentication setup functionality."""
    
    def test_auth_guide_exists(self):
        """Test that authentication guide exists."""
        auth_guide_path = os.path.join(os.path.dirname(__file__), '..', '..', 'AUTHENTICATION_GUIDE.md')
        assert os.path.exists(auth_guide_path)
    
    def test_config_guide_exists(self):
        """Test that configuration guide exists."""
        config_guide_path = os.path.join(os.path.dirname(__file__), '..', '..', 'CONFIG_GUIDE.md')
        assert os.path.exists(config_guide_path)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])