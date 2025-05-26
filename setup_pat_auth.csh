#!/bin/csh
echo "🔐 Setting up Personal Access Token Authentication for All Accounts"
echo "=================================================================="
echo ""

# Function to setup PAT authentication for a specific account
echo "Setting up authentication for each GitHub account..."
echo ""

# Clear any existing credentials that might be causing issues
echo "🧹 Clearing existing credentials..."
if (-f ~/.git-credentials) then
    rm ~/.git-credentials
    echo "✅ Removed existing credential file"
endif

# Make sure credential helper is configured
git config --global credential.helper store
echo "✅ Configured credential helper"

echo ""
echo "Now we'll test authentication for each account:"
echo ""

# Test SanjeevaRDodlapati (should already work)
echo "🔑 Testing SanjeevaRDodlapati account..."
git ls-remote https://github.com/SanjeevaRDodlapati/UAVarPrior.git >/dev/null 2>&1
if ($status == 0) then
    echo "✅ SanjeevaRDodlapati authentication working"
else
    echo "❌ SanjeevaRDodlapati needs authentication"
    echo "Please enter credentials when prompted..."
    git ls-remote https://github.com/SanjeevaRDodlapati/UAVarPrior.git
endif

echo ""

# Test sdodlapati3
echo "🔑 Testing sdodlapati3 account..."
echo "For username, enter: sdodlapati3"
echo "For password, enter your Personal Access Token"
git ls-remote https://github.com/sdodlapati3/UAVarPrior.git
if ($status == 0) then
    echo "✅ sdodlapati3 authentication successful"
else
    echo "❌ sdodlapati3 authentication failed"
endif

echo ""

# Test sdodlapa
echo "🔑 Testing sdodlapa account..."
echo "For username, enter: sdodlapa"
echo "For password, enter your Personal Access Token: ghp_XdNR3nYlLnvzPu7wz5kpUCBhZwqFMJ2KF6k4"
git ls-remote https://github.com/sdodlapa/UAVarPrior.git
if ($status == 0) then
    echo "✅ sdodlapa authentication successful"
else
    echo "❌ sdodlapa authentication failed"
endif

echo ""
echo "🎯 Authentication setup complete!"
echo "📋 Stored credentials in ~/.git-credentials:"
if (-f ~/.git-credentials) then
    echo "Credential file exists with $(wc -l < ~/.git-credentials) entries"
else
    echo "⚠️  No credential file created - authentication may have failed"
endif

echo ""
echo "Now you can run ./push_all.csh to push to all accounts!"
