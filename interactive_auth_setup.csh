#!/bin/tcsh
# Interactive authentication setup for all accounts

echo "🔐 Setting up authentication for all GitHub accounts"
echo "=================================================="
echo ""

echo "You mentioned you have access tokens for all three accounts."
echo "Let's test them one by one."
echo ""

echo "📁 Current repository: `basename $PWD`"
echo ""

# Test sdodlapati3 first
echo "🔑 Testing sdodlapati3 account:"
echo "================================"
echo "When prompted, enter:"
echo "  Username: sdodlapati3"
echo "  Password: [Your Personal Access Token for sdodlapati3]"
echo ""

read -p "Press Enter when ready to test sdodlapati3..."
echo ""

echo "🚀 Pushing to sdodlapati3..."
git push sdodlapati3 main

if ($status == 0) then
    echo "✅ SUCCESS! sdodlapati3 authentication working!"
    set auth1_success = 1
else
    echo "❌ Authentication failed for sdodlapati3"
    set auth1_success = 0
endif

echo ""
echo "🔑 Testing sdodlapa account:"
echo "==========================="
echo "When prompted, enter:"
echo "  Username: sdodlapa"
echo "  Password: [Your Personal Access Token for sdodlapa]"
echo ""

read -p "Press Enter when ready to test sdodlapa..."
echo ""

echo "🚀 Pushing to sdodlapa..."
git push sdodlapa main

if ($status == 0) then
    echo "✅ SUCCESS! sdodlapa authentication working!"
    set auth2_success = 1
else
    echo "❌ Authentication failed for sdodlapa"
    set auth2_success = 0
endif

echo ""
echo "📊 FINAL RESULTS:"
echo "================"
echo "SanjeevaRDodlapati (origin): ✅ Working"

if ($auth1_success == 1) then
    echo "sdodlapati3: ✅ Working"
else
    echo "sdodlapati3: ❌ Needs setup"
endif

if ($auth2_success == 1) then
    echo "sdodlapa: ✅ Working"
else
    echo "sdodlapa: ❌ Needs setup"
endif

echo ""

if ($auth1_success == 1 && $auth2_success == 1) then
    echo "🎉 ALL ACCOUNTS WORKING!"
    echo "You can now use: ./push_all.csh"
    echo ""
    echo "🚀 Testing the full push_all script:"
    ./push_all.csh
else
    echo "⚠️  Some accounts still need setup."
    echo ""
    echo "💡 Troubleshooting tips:"
    echo "- Make sure repositories exist on GitHub"
    echo "- Use Personal Access Token (not GitHub password)"
    echo "- Token must have 'repo' scope"
    echo "- Username must be exact (case-sensitive)"
endif
