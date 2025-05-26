#!/usr/bin/env python
# UAVarPrior Dependency Report Generator
# This script outputs a detailed report of all installed packages and their versions
# to help diagnose compatibility issues

import os
import sys
import subprocess
import json
from datetime import datetime

def print_separator(char="-", length=60):
    print(char * length)

def get_pip_list():
    """Get list of installed packages using pip"""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "list", "--format=json"],
            capture_output=True,
            text=True,
            check=True
        )
        return json.loads(result.stdout)
    except Exception as e:
        print(f"Error getting pip list: {e}")
        return []

def get_package_details(package_name):
    """Get detailed information about a specific package"""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", package_name],
            capture_output=True,
            text=True,
            check=True
        )
        
        details = {}
        current_key = None
        
        for line in result.stdout.splitlines():
            if not line.strip():
                continue
                
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()
                details[key] = value
                current_key = key
            elif current_key:
                details[current_key] += "\n" + line.strip()
                
        return details
    except Exception as e:
        print(f"Error getting details for {package_name}: {e}")
        return {}

def check_key_dependencies():
    """Check for specific key dependencies required by UAVarPrior"""
    key_deps = {
        "numpy": "1.19.0-2.0.0",
        "cython": "0.29.0-3.0.0",
        "torch": "1.10.0-2.0.0",
        "scipy": "1.7.0-1.10.0",
        "pandas": "1.3.0-2.0.0",
        "matplotlib": "3.4.0-3.7.0",
        "scikit-learn": "1.0.0-1.3.0",
        "biopython": "1.79-1.82",
        "h5py": "3.1.0-4.0.0",
        "pyyaml": "5.1-6.0"
    }
    
    results = {}
    for dep, version_range in key_deps.items():
        try:
            module = __import__(dep)
            version = getattr(module, "__version__", "unknown")
            results[dep] = {
                "installed": True,
                "version": version,
                "recommended_range": version_range
            }
        except ImportError:
            results[dep] = {
                "installed": False,
                "recommended_range": version_range
            }
    
    return results

def check_uavarprior():
    """Check UAVarPrior package installation"""
    try:
        import uavarprior
        version = getattr(uavarprior, "__version__", "unknown")
        path = os.path.dirname(uavarprior.__file__)
        
        result = {
            "installed": True,
            "version": version,
            "path": path
        }
        
        # Check for Cython modules
        try:
            from uavarprior.data.sequences import _sequence
            result["_sequence_module"] = "OK"
        except ImportError as e:
            result["_sequence_module"] = f"ERROR: {str(e)}"
        
        try:
            from uavarprior.data.targets import _genomic_features
            result["_genomic_features_module"] = "OK"
        except ImportError as e:
            result["_genomic_features_module"] = f"ERROR: {str(e)}"
            
        return result
    except ImportError:
        return {
            "installed": False
        }

def collect_sys_info():
    """Collect system information"""
    info = {
        "python_version": sys.version,
        "python_path": sys.executable,
        "platform": sys.platform,
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Try to get additional environment information
    try:
        info["cpu_info"] = subprocess.run(
            ["cat", "/proc/cpuinfo"], capture_output=True, text=True
        ).stdout.split("\n\n")[0]
    except:
        pass
    
    try:
        info["memory_info"] = subprocess.run(
            ["free", "-h"], capture_output=True, text=True
        ).stdout
    except:
        pass
        
    return info

def generate_report():
    """Generate the full dependency report"""
    print_separator("=")
    print(f"UAVarPrior DEPENDENCY REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print_separator("=")
    
    # System information
    sys_info = collect_sys_info()
    print("SYSTEM INFORMATION:")
    for key, value in sys_info.items():
        if key in ["cpu_info", "memory_info"]:
            print(f"{key}:\n{value}")
        else:
            print(f"{key}: {value}")
    print_separator()
    
    # UAVarPrior package
    uavarprior_info = check_uavarprior()
    print("UAVARPRIOR PACKAGE:")
    if uavarprior_info["installed"]:
        print(f"Installed: Yes")
        print(f"Version: {uavarprior_info['version']}")
        print(f"Path: {uavarprior_info['path']}")
        print(f"_sequence module: {uavarprior_info['_sequence_module']}")
        print(f"_genomic_features module: {uavarprior_info['_genomic_features_module']}")
    else:
        print("Installed: No")
    print_separator()
    
    # Key dependencies
    key_deps = check_key_dependencies()
    print("KEY DEPENDENCIES:")
    for package, info in key_deps.items():
        status = "✅ Installed" if info["installed"] else "❌ Missing"
        version = info.get("version", "N/A")
        rec_range = info["recommended_range"]
        print(f"{package}: {status} (v{version}, recommended: {rec_range})")
    print_separator()
    
    # All installed packages
    print("ALL INSTALLED PACKAGES:")
    packages = get_pip_list()
    packages.sort(key=lambda x: x["name"].lower())
    for pkg in packages:
        name = pkg["name"]
        version = pkg["version"]
        print(f"{name}: {version}")
    print_separator()
    
    # Save to file
    report_path = "dependency_report.txt"
    try:
        with open(report_path, "w") as f:
            # Redirect stdout to file
            old_stdout = sys.stdout
            sys.stdout = f
            
            print_separator("=")
            print(f"UAVarPrior DEPENDENCY REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print_separator("=")
            
            # System information
            print("SYSTEM INFORMATION:")
            for key, value in sys_info.items():
                if key in ["cpu_info", "memory_info"]:
                    print(f"{key}:\n{value}")
                else:
                    print(f"{key}: {value}")
            print_separator()
            
            # UAVarPrior package
            print("UAVARPRIOR PACKAGE:")
            if uavarprior_info["installed"]:
                print(f"Installed: Yes")
                print(f"Version: {uavarprior_info['version']}")
                print(f"Path: {uavarprior_info['path']}")
                print(f"_sequence module: {uavarprior_info['_sequence_module']}")
                print(f"_genomic_features module: {uavarprior_info['_genomic_features_module']}")
            else:
                print("Installed: No")
            print_separator()
            
            # Key dependencies
            print("KEY DEPENDENCIES:")
            for package, info in key_deps.items():
                status = "Installed" if info["installed"] else "Missing"
                version = info.get("version", "N/A")
                rec_range = info["recommended_range"]
                print(f"{package}: {status} (v{version}, recommended: {rec_range})")
            print_separator()
            
            # All installed packages
            print("ALL INSTALLED PACKAGES:")
            for pkg in packages:
                name = pkg["name"]
                version = pkg["version"]
                print(f"{name}: {version}")
                
                # Include detailed info for key packages
                if name.lower() in [p.lower() for p in key_deps.keys()] or name.lower() == "uavarprior":
                    details = get_package_details(name)
                    if "Requires" in details:
                        print(f"  Requires: {details['Requires']}")
                    if "Required-by" in details:
                        print(f"  Required-by: {details['Required-by']}")
                    print()
            
            # Restore stdout
            sys.stdout = old_stdout
        
        print(f"Detailed dependency report saved to {report_path}")
    except Exception as e:
        print(f"Error saving report to file: {e}")

if __name__ == "__main__":
    generate_report()
