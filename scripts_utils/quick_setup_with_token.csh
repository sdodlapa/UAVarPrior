#!/bin/csh
echo "🚀 Quick Setup with Valid Personal Access Tokens"
echo "================================================"
echo ""

# IMPORTANT: Replace these placeholder tokens with your actual tokens
set TOKEN_SANJEEVA = "YOUR_SANJEEVA_TOKEN_HERE"
set TOKEN_SDODLAPATI3 = "YOUR_SDODLAPATI3_TOKEN_HERE"  
set TOKEN_SDODLAPA = "YOUR_SDODLAPA_TOKEN_HERE"

echo "⚠️  EDIT THIS SCRIPT FIRST!"
echo "=========================="
echo ""
echo "Before running this script, you need to:"
echo "1. Edit this file and replace the placeholder tokens with real ones"
echo "2. Make sure all repositories exist (run ./create_missing_repos.csh for guidance)"
echo ""

# Check if tokens have been updated
if ("$TOKEN_SANJEEVA" == "YOUR_SANJEEVA_TOKEN_HERE") then
    echo "❌ Please edit this script and add your actual Personal Access Tokens"
    echo ""
    echo "To edit: nano quick_setup_with_token.csh"
    echo "Replace YOUR_*_TOKEN_HERE with actual tokens from GitHub"
    exit 1
endif

echo "🧹 Clearing existing credentials..."
if (-f ~/.git-credentials) then
    rm ~/.git-credentials
endif

echo "🔧 Setting up credential storage..."
git config --global credential.helper store

echo "📝 Creating credential entries..."
cat > ~/.git-credentials << EOF
https://SanjeevaRDodlapati:${TOKEN_SANJEEVA}@github.com
https://sdodlapati3:${TOKEN_SDODLAPATI3}@github.com
https://sdodlapa:${TOKEN_SDODLAPA}@github.com
EOF

chmod 600 ~/.git-credentials
echo "✅ Credential file created with secure permissions"

echo ""
echo "🔧 Updating remote URLs for UAVarPrior..."
git remote set-url origin https://SanjeevaRDodlapati:${TOKEN_SANJEEVA}@github.com/SanjeevaRDodlapati/UAVarPrior.git
git remote set-url sdodlapati3 https://sdodlapati3:${TOKEN_SDODLAPATI3}@github.com/sdodlapati3/UAVarPrior.git
git remote set-url sdodlapa https://sdodlapa:${TOKEN_SDODLAPA}@github.com/sdodlapa/UAVarPrior.git

echo "✅ Updated all remote URLs"

echo ""
echo "🧪 Testing authentication..."

echo "Testing SanjeevaRDodlapati..."
curl -s -H "Authorization: token ${TOKEN_SANJEEVA}" https://api.github.com/user | grep -q '"login"'
if ($status == 0) then
    echo "✅ SanjeevaRDodlapati token valid"
else
    echo "❌ SanjeevaRDodlapati token invalid"
endif

echo "Testing sdodlapati3..."
curl -s -H "Authorization: token ${TOKEN_SDODLAPATI3}" https://api.github.com/user | grep -q '"login"'
if ($status == 0) then
    echo "✅ sdodlapati3 token valid"
else
    echo "❌ sdodlapati3 token invalid"
endif

echo "Testing sdodlapa..."
curl -s -H "Authorization: token ${TOKEN_SDODLAPA}" https://api.github.com/user | grep -q '"login"'
if ($status == 0) then
    echo "✅ sdodlapa token valid"
else
    echo "❌ sdodlapa token invalid"
endif

echo ""
echo "🚀 Testing push operations..."

echo "Testing push to SanjeevaRDodlapati..."
git push origin main
if ($status == 0) then
    echo "✅ Push to SanjeevaRDodlapati successful"
else
    echo "❌ Push to SanjeevaRDodlapati failed"
endif

echo ""
echo "Testing push to sdodlapati3..."
git push sdodlapati3 main
if ($status == 0) then
    echo "✅ Push to sdodlapati3 successful"
else
    echo "❌ Push to sdodlapati3 failed"
endif

echo ""
echo "Testing push to sdodlapa..."
git push sdodlapa main
if ($status == 0) then
    echo "✅ Push to sdodlapa successful"
else
    echo "❌ Push to sdodlapa failed"
endif

echo ""
echo "🎯 Setup Complete!"
echo "=================="
echo ""
echo "If all tests passed, you can now run:"
echo "  ./push_all.csh"
echo ""
echo "To set up other repositories, copy this pattern to:"
echo "  /home/sdodl001/GenomicLightning/"
echo "  /home/sdodl001/FuGEP/"
