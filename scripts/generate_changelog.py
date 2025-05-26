#!/usr/bin/env python3
"""
Generate changelog entries for UAVarPrior releases.
"""

import sys
import argparse
from version_manager import UAVarPriorVersionManager

def main():
    """Generate changelog entry."""
    parser = argparse.ArgumentParser(description="Generate UAVarPrior changelog")
    parser.add_argument("version", help="Version to generate changelog for")
    parser.add_argument("--update", action="store_true",
                       help="Update CHANGELOG.md file")
    
    args = parser.parse_args()
    
    try:
        manager = UAVarPriorVersionManager()
        entry = manager.generate_changelog_entry(args.version)
        
        if args.update:
            manager.update_changelog(args.version)
            print(f"Changelog updated for version {args.version}")
        else:
            print(entry)
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()