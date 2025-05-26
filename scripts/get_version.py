#!/usr/bin/env python3
"""
Simple script to get the current version of UAVarPrior.
"""

import sys
from version_manager import UAVarPriorVersionManager

def main():
    """Get and print the current version."""
    try:
        manager = UAVarPriorVersionManager()
        version = manager.get_current_version()
        print(version)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()