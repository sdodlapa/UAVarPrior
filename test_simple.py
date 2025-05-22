try:
    import uavarprior
    print(f'SUCCESS: UAVarPrior version {uavarprior.__version__} imported successfully!')
except Exception as e:
    print(f'ERROR: {e}')
