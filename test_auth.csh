#!/bin/tcsh
# Interactive authentication test for sdodlapati3 account

echo "🔐 Testing authentication for sdodlapati3 account"
echo "==============================================="
echo ""

echo "This will test pushing to your second GitHub account."
echo "When prompted for credentials:"
echo ""
echo "Username: sdodlapati3"
echo "Password: [Your Personal Access Token]"
echo ""
echo "The token will be stored for future use."
echo ""

echo -n "Press Enter when you have your Personal Access Token ready..."
set dummy = $<

echo ""
echo "🚀 Testing push to sdodlapati3/UAVarPrior..."
git push sdodlapati3 main

if ($status == 0) then
    echo ""
    echo "🎉 SUCCESS! Authentication configured!"
    echo "✅ Your credentials are now stored"
    echo "✅ Future pushes will be automatic"
    echo ""
    echo "You can now use: ./push_all.csh"
else
    echo ""
    echo "❌ Authentication failed"
    echo "Please check:"
    echo "  1. Username is exactly: sdodlapati3"
    echo "  2. Password is your Personal Access Token (not your GitHub password)"
    echo "  3. Token has 'repo' scope permissions"
endif
