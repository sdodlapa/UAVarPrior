#!/usr/bin/env python
# Final verification test for UAVarPrior installation

print("=" * 60)
print("UAVarPrior Installation Verification Test")
print("=" * 60)

# Test 1: Basic package import
print("\n1. Testing basic package import...")
try:
    import uavarprior
    print("✅ SUCCESS: UAVarPrior imported successfully!")
    version = getattr(uavarprior, '__version__', 'unknown')
    print(f"   Package version: {version}")
except Exception as e:
    print(f"❌ FAILED: {e}")
    exit(1)

# Test 2: Test key dependencies
print("\n2. Testing key dependencies...")
deps = [
    ('numpy', 'NumPy'),
    ('torch', 'PyTorch'),
    ('pandas', 'Pandas'),
    ('scipy', 'SciPy'),
    ('matplotlib', 'Matplotlib'),
    ('sklearn', 'Scikit-learn'),
    ('h5py', 'HDF5'),
    ('Bio', 'BioPython')
]

for module, name in deps:
    try:
        mod = __import__(module)
        version = getattr(mod, '__version__', 'unknown')
        print(f"✅ {name}: {version}")
    except ImportError:
        print(f"❌ {name}: Not found")

# Test 3: Test UAVarPrior submodules
print("\n3. Testing UAVarPrior submodules...")
submodules = [
    'uavarprior.data',
    'uavarprior.data.sequences', 
    'uavarprior.data.targets',
    'uavarprior.samplers',
    'uavarprior.interpret'
]

for module in submodules:
    try:
        __import__(module)
        print(f"✅ {module}: OK")
    except ImportError as e:
        print(f"❌ {module}: {e}")

# Test 4: Test Cython extensions
print("\n4. Testing Cython extension modules...")
try:
    from uavarprior.data.sequences import _sequence
    print("✅ _sequence extension: OK")
except ImportError as e:
    print(f"❌ _sequence extension: {e}")

try:
    from uavarprior.data.targets import _genomic_features
    print("✅ _genomic_features extension: OK")
except ImportError as e:
    print(f"❌ _genomic_features extension: {e}")

# Test 5: Test specific classes
print("\n5. Testing specific UAVarPrior classes...")
try:
    from uavarprior.data.sequences import Sequence, Genome
    print("✅ Sequence classes: OK")
except ImportError as e:
    print(f"❌ Sequence classes: {e}")

try:
    from uavarprior.data.targets import Target, GenomicFeatures
    print("✅ Target classes: OK")
except ImportError as e:
    print(f"❌ Target classes: {e}")

print("\n" + "=" * 60)
print("Verification test completed!")
print("If you see mostly ✅ marks above, the installation is successful.")
print("=" * 60)
