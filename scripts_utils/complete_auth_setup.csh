#!/bin/csh
echo "🔧 Complete GitHub Authentication Setup Guide"
echo "============================================="
echo ""
echo "The current Personal Access Token appears to be invalid/expired."
echo "Let's set up fresh authentication for all three accounts."
echo ""

echo "📝 STEP 1: Generate Personal Access Tokens"
echo "=========================================="
echo ""
echo "You need to create Personal Access Tokens for each account:"
echo ""
echo "For SanjeevaRDodlapati account:"
echo "  1. Go to: https://github.com/settings/tokens"
echo "  2. Click 'Generate new token (classic)'"
echo "  3. Give it a name like 'Multi-repo automation'"
echo "  4. Select scopes: repo (full control)"
echo "  5. Copy the generated token"
echo ""
echo "For sdodlapati3 account:"
echo "  1. Switch to sdodlapati3 account"
echo "  2. Go to: https://github.com/settings/tokens"
echo "  3. Create token with repo scope"
echo ""
echo "For sdodlapa account:"
echo "  1. Switch to sdodlapa account"
echo "  2. Go to: https://github.com/settings/tokens"
echo "  3. Create token with repo scope"
echo ""

echo "🔐 STEP 2: Interactive Token Setup"
echo "=================================="
echo ""

# Clear existing credentials
if (-f ~/.git-credentials) then
    rm ~/.git-credentials
    echo "🧹 Cleared existing credentials"
endif

git config --global credential.helper store
echo "✅ Configured credential storage"

echo ""
echo "Now let's set up each account..."
echo ""

echo "🔑 Setting up SanjeevaRDodlapati..."
echo "Enter Personal Access Token for SanjeevaRDodlapati:"
echo -n "Token: "
set TOKEN1 = $<
echo "https://SanjeevaRDodlapati:${TOKEN1}@github.com" >> ~/.git-credentials

echo ""
echo "🔑 Setting up sdodlapati3..."
echo "Enter Personal Access Token for sdodlapati3:"
echo -n "Token: "
set TOKEN2 = $<
echo "https://sdodlapati3:${TOKEN2}@github.com" >> ~/.git-credentials

echo ""
echo "🔑 Setting up sdodlapa..."
echo "Enter Personal Access Token for sdodlapa:"
echo -n "Token: "
set TOKEN3 = $<
echo "https://sdodlapa:${TOKEN3}@github.com" >> ~/.git-credentials

echo ""
chmod 600 ~/.git-credentials
echo "🔒 Set secure permissions on credential file"

echo ""
echo "🧪 STEP 3: Testing Authentication"
echo "================================="
echo ""

echo "Testing SanjeevaRDodlapati..."
curl -s -H "Authorization: token ${TOKEN1}" https://api.github.com/user | grep -q '"login"' 
if ($status == 0) then
    echo "✅ SanjeevaRDodlapati token is valid"
else
    echo "❌ SanjeevaRDodlapati token is invalid"
endif

echo "Testing sdodlapati3..."
curl -s -H "Authorization: token ${TOKEN2}" https://api.github.com/user | grep -q '"login"'
if ($status == 0) then
    echo "✅ sdodlapati3 token is valid"
else
    echo "❌ sdodlapati3 token is invalid"
endif

echo "Testing sdodlapa..."
curl -s -H "Authorization: token ${TOKEN3}" https://api.github.com/user | grep -q '"login"'
if ($status == 0) then
    echo "✅ sdodlapa token is valid"
else
    echo "❌ sdodlapa token is invalid"
endif

echo ""
echo "🔧 STEP 4: Update Remote URLs"
echo "============================="
echo ""

# Update remote URLs with tokens
git remote set-url origin https://SanjeevaRDodlapati:${TOKEN1}@github.com/SanjeevaRDodlapati/UAVarPrior.git
git remote set-url sdodlapati3 https://sdodlapati3:${TOKEN2}@github.com/sdodlapati3/UAVarPrior.git
git remote set-url sdodlapa https://sdodlapa:${TOKEN3}@github.com/sdodlapa/UAVarPrior.git

echo "✅ Updated all remote URLs with authentication tokens"
echo ""

echo "🚀 STEP 5: Test Push Operations"
echo "==============================="
echo ""

echo "Testing push to SanjeevaRDodlapati..."
git push origin main 2>&1 | head -5
echo ""

echo "Testing push to sdodlapati3..."
git push sdodlapati3 main 2>&1 | head -5
echo ""

echo "Testing push to sdodlapa..."
git push sdodlapa main 2>&1 | head -5
echo ""

echo "🎯 Setup Complete!"
echo "=================="
echo ""
echo "If all tests passed, you can now run:"
echo "  ./push_all.csh"
echo ""
echo "To push to all repositories at once!"
