from typing import Dict, Any
import importlib

def get_model(config: Dict[str, Any]):
    """Factory function for model instantiation.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        Instantiated model
    """
    model_name = config.get("name")
    if not model_name:
        raise ValueError("Model name not specified in configuration")
        
    try:
        # Import the appropriate model module
        module_path = f"uavarprior.models.{model_name}"
        module = importlib.import_module(module_path)
        
        # Get the model class and instantiate it
        model_class = getattr(module, "Model")  # Assuming each model module has a "Model" class
        return model_class(**config)
        
    except ImportError:
        raise ValueError(f"Model module not found: {model_name}")
    except AttributeError:
        raise ValueError(f"Model class not found in module: {model_name}")