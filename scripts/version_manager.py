#!/usr/bin/env python3
"""
Version management utilities for UAVarPrior.
Handles version bumping, changelog generation, and release preparation.
"""

import re
import sys
import argparse
from pathlib import Path
from typing import Optional, Tuple, Union
import subprocess
import json
from datetime import datetime


class UAVarPriorVersionManager:
    """Manage versions for the UAVarPrior project."""
    
    def __init__(self, root_dir: Optional[Union[str, Path]] = None):
        if root_dir is None:
            self.root_dir = Path(__file__).parent.parent
        else:
            self.root_dir = Path(root_dir) if isinstance(root_dir, str) else root_dir
        self.setup_py = self.root_dir / "setup.py"
        self.pyproject_toml = self.root_dir / "pyproject.toml"
        self.init_file = self.root_dir / "uavarprior" / "__init__.py"
        
    def get_current_version(self) -> str:
        """Get the current version from setup.py or pyproject.toml."""
        # Try pyproject.toml first
        if self.pyproject_toml.exists():
            content = self.pyproject_toml.read_text()
            match = re.search(r'version\s*=\s*["\']([^"\']+)["\']', content)
            if match:
                return match.group(1)
        
        # Fall back to setup.py
        if self.setup_py.exists():
            content = self.setup_py.read_text()
            match = re.search(r'version\s*=\s*["\']([^"\']+)["\']', content)
            if match:
                return match.group(1)
        
        # Try __init__.py
        if self.init_file.exists():
            content = self.init_file.read_text()
            match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
            if match:
                return match.group(1)
        
        # Default fallback version
        return "0.1.0"
    
    def parse_version(self, version: str) -> Tuple[int, int, int]:
        """Parse semantic version string into components."""
        match = re.match(r'(\d+)\.(\d+)\.(\d+)', version)
        if not match:
            raise ValueError(f"Invalid version format: {version}")
        return tuple(map(int, match.groups()))
    
    def bump_version(self, bump_type: str = "patch") -> str:
        """Bump version according to semantic versioning."""
        current = self.get_current_version()
        major, minor, patch = self.parse_version(current)
        
        if bump_type == "major":
            major += 1
            minor = 0
            patch = 0
        elif bump_type == "minor":
            minor += 1
            patch = 0
        elif bump_type == "patch":
            patch += 1
        else:
            raise ValueError(f"Invalid bump type: {bump_type}")
        
        new_version = f"{major}.{minor}.{patch}"
        self._update_version_files(new_version)
        return new_version
    
    def _update_version_files(self, new_version: str):
        """Update version in all relevant files."""
        # Update pyproject.toml
        if self.pyproject_toml.exists():
            content = self.pyproject_toml.read_text()
            content = re.sub(
                r'version\s*=\s*["\'][^"\']+["\']',
                f'version = "{new_version}"',
                content
            )
            self.pyproject_toml.write_text(content)
        
        # Update setup.py
        if self.setup_py.exists():
            content = self.setup_py.read_text()
            content = re.sub(
                r'version\s*=\s*["\'][^"\']+["\']',
                f'version="{new_version}"',
                content
            )
            self.setup_py.write_text(content)
        
        # Update __init__.py
        if self.init_file.exists():
            content = self.init_file.read_text()
            if '__version__' in content:
                content = re.sub(
                    r'__version__\s*=\s*["\'][^"\']+["\']',
                    f'__version__ = "{new_version}"',
                    content
                )
            else:
                # Add version if it doesn't exist
                content = f'__version__ = "{new_version}"\n' + content
            self.init_file.write_text(content)
    
    def generate_changelog_entry(self, version: str) -> str:
        """Generate changelog entry for the new version."""
        try:
            # Get commits since last tag
            result = subprocess.run(
                ["git", "log", "--oneline", "--since=1 week ago"],
                capture_output=True,
                text=True,
                cwd=self.root_dir
            )
            commits = result.stdout.strip().split('\n') if result.stdout.strip() else []
            
            date_str = datetime.now().strftime("%Y-%m-%d")
            changelog = f"\n## [{version}] - {date_str}\n\n"
            
            if commits:
                changelog += "### Changes\n"
                for commit in commits[:10]:  # Limit to 10 recent commits
                    if commit.strip():
                        changelog += f"- {commit.strip()}\n"
            else:
                changelog += "### Changes\n- Version bump and improvements\n"
            
            return changelog
            
        except subprocess.CalledProcessError:
            # Fallback if git is not available
            date_str = datetime.now().strftime("%Y-%m-%d")
            return f"\n## [{version}] - {date_str}\n\n### Changes\n- Version bump and improvements\n"
    
    def update_changelog(self, version: str):
        """Update CHANGELOG.md with new version entry."""
        changelog_file = self.root_dir / "CHANGELOG.md"
        entry = self.generate_changelog_entry(version)
        
        if changelog_file.exists():
            content = changelog_file.read_text()
            # Insert new entry after the header
            lines = content.split('\n')
            header_end = 0
            for i, line in enumerate(lines):
                if line.startswith('## [') or line.startswith('### '):
                    header_end = i
                    break
            
            lines.insert(header_end, entry.rstrip())
            changelog_file.write_text('\n'.join(lines))
        else:
            # Create new changelog
            header = "# Changelog\n\nAll notable changes to this project will be documented in this file.\n"
            changelog_file.write_text(header + entry)
    
    def create_git_tag(self, version: str, push: bool = False):
        """Create and optionally push a git tag."""
        try:
            subprocess.run(
                ["git", "tag", "-a", f"v{version}", "-m", f"Release version {version}"],
                check=True,
                cwd=self.root_dir
            )
            print(f"Created git tag: v{version}")
            
            if push:
                subprocess.run(
                    ["git", "push", "origin", f"v{version}"],
                    check=True,
                    cwd=self.root_dir
                )
                print(f"Pushed git tag: v{version}")
                
        except subprocess.CalledProcessError as e:
            print(f"Git operation failed: {e}")
    
    def prepare_release(self, bump_type: str = "patch", push_tag: bool = False) -> str:
        """Complete release preparation workflow."""
        print(f"Preparing UAVarPrior release with {bump_type} version bump...")
        
        # Bump version
        new_version = self.bump_version(bump_type)
        print(f"Version bumped to: {new_version}")
        
        # Update changelog
        self.update_changelog(new_version)
        print("Changelog updated")
        
        # Create git tag
        self.create_git_tag(new_version, push_tag)
        
        print(f"UAVarPrior release {new_version} prepared successfully!")
        return new_version


def main():
    """CLI interface for version management."""
    parser = argparse.ArgumentParser(description="UAVarPrior Version Manager")
    parser.add_argument("action", choices=["get", "bump", "release"],
                       help="Action to perform")
    parser.add_argument("--type", choices=["major", "minor", "patch"],
                       default="patch", help="Type of version bump")
    parser.add_argument("--push-tag", action="store_true",
                       help="Push git tag to remote")
    
    args = parser.parse_args()
    
    manager = UAVarPriorVersionManager()
    
    try:
        if args.action == "get":
            version = manager.get_current_version()
            print(f"Current UAVarPrior version: {version}")
        
        elif args.action == "bump":
            new_version = manager.bump_version(args.type)
            print(f"Version bumped to: {new_version}")
        
        elif args.action == "release":
            new_version = manager.prepare_release(args.type, args.push_tag)
            print(f"Release {new_version} prepared")
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()