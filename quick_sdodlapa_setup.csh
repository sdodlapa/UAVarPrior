#!/bin/tcsh
# Quick setup after sdodlapa repository is created

echo "🚀 Quick Setup for sdodlapa Repository"
echo "====================================="
echo ""

echo "This script assumes you've created the UAVarPrior repository on:"
echo "https://github.com/sdodlapa/UAVarPrior"
echo ""

echo "Testing repository existence..."
git ls-remote sdodlapa HEAD >& /dev/null

if ($status == 0) then
    echo "✅ Repository found!"
    echo ""
    echo "🔑 Setting up authentication..."
    echo "Enter your sdodlapa Personal Access Token when prompted:"
    
    git config credential.https://github.com/sdodlapa.username sdodlapa
    git push sdodlapa main
    
    if ($status == 0) then
        echo ""
        echo "🎉 SUCCESS! sdodlapa account configured!"
        echo ""
        echo "🚀 Now testing complete push to all accounts:"
        ./push_all.csh
    else
        echo ""
        echo "❌ Authentication failed"
        echo "Make sure you use your Personal Access Token as password"
    endif
else
    echo "❌ Repository not found"
    echo ""
    echo "Please create the repository first:"
    echo "1. Go to: https://github.com/sdodlapa"
    echo "2. Click 'New repository'"
    echo "3. Name: UAVarPrior"
    echo "4. Public repository"
    echo "5. DO NOT add README"
    echo "6. Click 'Create repository'"
    echo ""
    echo "Then run this script again: ./quick_sdodlapa_setup.csh"
endif
