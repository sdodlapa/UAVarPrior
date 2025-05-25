#!/bin/tcsh
# Quick setup script for GitHub authentication

echo "🔐 GitHub Authentication Setup"
echo "=============================="
echo ""

echo "Setting up authentication for multiple GitHub accounts..."
echo ""

echo "Step 1: Configure credential storage"
git config --global credential.helper store
echo "✅ Credential helper configured"
echo ""

echo "Step 2: Check current remotes"
git remote -v
echo ""

echo "Step 3: Create repository on second account"
echo "📝 Action required:"
echo "   1. Go to: https://github.com/sdodlapati3"
echo "   2. Click 'New repository'"
echo "   3. Name: UAVarPrior"
echo "   4. Make it Public (or Private if you prefer)"
echo "   5. DO NOT initialize with README (since you're pushing existing code)"
echo "   6. Click 'Create repository'"
echo ""

echo "Step 4: Generate Personal Access Token"
echo "📝 Action required:"
echo "   1. Go to: https://github.com/settings/tokens"
echo "   2. Click 'Generate new token (classic)'"
echo "   3. Name: 'UAVarPrior-access' or similar"
echo "   4. Scopes: Check 'repo' (full repository access)"
echo "   5. Click 'Generate token'"
echo "   6. COPY THE TOKEN (you won't see it again!)"
echo ""

echo "Step 5: Test authentication"
echo "Run this command to test:"
echo "   git push secondary main"
echo ""
echo "When prompted:"
echo "   Username: sdodlapati3"
echo "   Password: [paste your Personal Access Token]"
echo ""

echo "The credentials will be stored for future use."
echo ""
echo "🎯 After completing these steps, run:"
echo "   ./push_all.csh"
echo ""
echo "This will push to both accounts automatically!"
