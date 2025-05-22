import sys
import traceback

try:
    import uavarprior
    print(f"SUCCESS: UAVarPrior version {uavarprior.__version__} imported successfully!")
    print(f"Python path: {sys.path}")
except ImportError as e:
    print(f"IMPORT ERROR: {e}")
    print(f"Python path: {sys.path}")
except Exception as e:
    print(f"OTHER ERROR: {e}")
    traceback.print_exc()
    print(f"Python path: {sys.path}")
