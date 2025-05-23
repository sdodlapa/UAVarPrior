try:\n    from src.uavarprior.predict import PeakGVarEvaluator\n    print("Class found!")\nexcept ImportError as e:\n    print(f"Import error: {e}")
