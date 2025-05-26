#!/bin/tcsh
# Comprehensive authentication setup for all GitHub accounts

echo "🔐 Multi-Account GitHub Authentication Setup"
echo "==========================================="
echo ""

echo "Current stored credentials:"
if (-f ~/.git-credentials) then
    echo "📁 Found existing credentials file"
    cat ~/.git-credentials
    echo ""
else
    echo "📁 No stored credentials found"
    echo ""
endif

echo "We need to set up credentials for all three accounts:"
echo "1. SanjeevaRDodlapati (already working ✅)"
echo "2. sdodlapati3 (needs setup)"
echo "3. sdodlapa (needs setup)"
echo ""

echo "🎯 Setting up per-remote authentication..."
echo ""

# Set up credential storage for specific remotes
echo "Configuring credential helpers for each account..."

# For sdodlapati3
git config credential.https://github.com/sdodlapati3.username sdodlapati3
echo "✅ Set username for sdodlapati3"

# For sdodlapa  
git config credential.https://github.com/sdodlapa.username sdodlapa
echo "✅ Set username for sdodlapa"

echo ""
echo "🔑 Now testing authentication for each account..."
echo "You'll be prompted for passwords (use your Personal Access Tokens)"
echo ""

echo "Testing sdodlapati3..."
echo "When prompted for password, enter your sdodlapati3 Personal Access Token:"
git push sdodlapati3 main

set auth1_status = $status
echo ""

if ($auth1_status == 0) then
    echo "✅ sdodlapati3 authentication successful!"
else
    echo "❌ sdodlapati3 authentication failed"
    echo "Possible issues:"
    echo "- Repository doesn't exist on sdodlapati3 account"
    echo "- Wrong Personal Access Token"
    echo "- Token doesn't have 'repo' scope"
endif

echo ""
echo "Testing sdodlapa..."
echo "When prompted for password, enter your sdodlapa Personal Access Token:"
git push sdodlapa main

set auth2_status = $status
echo ""

if ($auth2_status == 0) then
    echo "✅ sdodlapa authentication successful!"
else
    echo "❌ sdodlapa authentication failed"
    echo "Possible issues:"
    echo "- Repository doesn't exist on sdodlapa account"
    echo "- Wrong Personal Access Token"  
    echo "- Token doesn't have 'repo' scope"
endif

echo ""
echo "📊 AUTHENTICATION SUMMARY"
echo "========================"
echo "SanjeevaRDodlapati: ✅ Working"

if ($auth1_status == 0) then
    echo "sdodlapati3: ✅ Working"
else
    echo "sdodlapati3: ❌ Failed"
endif

if ($auth2_status == 0) then
    echo "sdodlapa: ✅ Working"
else
    echo "sdodlapa: ❌ Failed"
endif

echo ""

if ($auth1_status == 0 && $auth2_status == 0) then
    echo "🎉 ALL ACCOUNTS AUTHENTICATED!"
    echo ""
    echo "🚀 Testing complete push_all script:"
    ./push_all.csh
else
    echo "⚠️ Some accounts need attention"
    echo ""
    echo "To create missing repositories:"
    if ($auth2_status != 0) then
        echo "• Go to https://github.com/sdodlapa and create repositories"
    endif
    echo ""
    echo "To fix authentication:"
    echo "• Ensure Personal Access Tokens have 'repo' scope"
    echo "• Use tokens as passwords, not GitHub passwords"
    echo "• Check repository names are exact matches"
endif
