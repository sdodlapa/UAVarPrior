try:
    from uavarprior.predict import PeakGVarEvaluator
    print("Class found!")
except ImportError as e:
    print(f"Import error: {e}")
