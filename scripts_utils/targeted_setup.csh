#!/bin/tcsh
# Targeted authentication and repository setup

echo "🎯 GitHub Multi-Account Setup Status"
echo "==================================="
echo ""

echo "📊 Repository Status Check:"
echo ""

# Check sdodlapati3
echo -n "sdodlapati3/UAVarPrior... "
git ls-remote sdodlapati3 HEAD >& /dev/null
if ($status == 0) then
    echo "✅ Repository exists"
    set repo1_exists = 1
else
    echo "❌ Repository not found"
    set repo1_exists = 0
endif

# Check sdodlapa  
echo -n "sdodlapa/UAVarPrior... "
git ls-remote sdodlapa HEAD >& /dev/null
if ($status == 0) then
    echo "✅ Repository exists"
    set repo2_exists = 1
else
    echo "❌ Repository not found"
    set repo2_exists = 0
endif

echo ""

if ($repo1_exists == 1) then
    echo "🔑 Setting up authentication for sdodlapati3..."
    echo "When prompted for password, enter your sdodlapati3 Personal Access Token:"
    echo ""
    
    # Clear any cached credentials for this specific repository
    git config --unset credential.https://github.com/sdodlapati3.helper
    git config credential.https://github.com/sdodlapati3.username sdodlapati3
    
    echo "🚀 Testing push to sdodlapati3..."
    git push sdodlapati3 main
    
    set auth1_status = $status
    
    if ($auth1_status == 0) then
        echo ""
        echo "✅ SUCCESS! sdodlapati3 authentication working!"
    else
        echo ""
        echo "❌ Authentication failed for sdodlapati3"
        echo "Make sure you use your Personal Access Token as the password"
    endif
else
    echo "⚠️ sdodlapati3 repository needs to be created first"
    set auth1_status = 1
endif

echo ""

if ($repo2_exists == 1) then
    echo "🔑 Setting up authentication for sdodlapa..."
    echo "When prompted for password, enter your sdodlapa Personal Access Token:"
    echo ""
    
    git config credential.https://github.com/sdodlapa.username sdodlapa
    
    echo "🚀 Testing push to sdodlapa..."
    git push sdodlapa main
    
    set auth2_status = $status
    
    if ($auth2_status == 0) then
        echo ""
        echo "✅ SUCCESS! sdodlapa authentication working!"
    else
        echo ""
        echo "❌ Authentication failed for sdodlapa"
        echo "Make sure you use your Personal Access Token as the password"
    endif
else
    echo "❌ sdodlapa repository doesn't exist"
    echo ""
    echo "📝 To create the repository:"
    echo "1. Go to: https://github.com/sdodlapa"
    echo "2. Click 'New repository'"
    echo "3. Name: UAVarPrior"
    echo "4. Make it Public"
    echo "5. DO NOT add README"
    echo "6. Click 'Create repository'"
    echo ""
    set auth2_status = 1
endif

echo ""
echo "📈 FINAL STATUS"
echo "=============="

if ($repo1_exists == 1 && $auth1_status == 0) then
    echo "sdodlapati3: ✅ Ready"
    set working_count = 1
else
    echo "sdodlapati3: ❌ Needs setup"
    set working_count = 0
endif

if ($repo2_exists == 1 && $auth2_status == 0) then
    echo "sdodlapa: ✅ Ready"
    @ working_count++
else
    echo "sdodlapa: ❌ Needs setup"
endif

echo "SanjeevaRDodlapati: ✅ Ready"
@ working_count++

echo ""

if ($working_count == 3) then
    echo "🎉 ALL ACCOUNTS READY!"
    echo ""
    echo "🚀 Testing complete push to all accounts:"
    ./push_all.csh
else
    echo "⚠️ $working_count/3 accounts ready"
    echo ""
    echo "Next steps:"
    if ($repo2_exists == 0) then
        echo "• Create repository on sdodlapa account"
    endif
    if ($repo1_exists == 1 && $auth1_status != 0) then
        echo "• Fix authentication for sdodlapati3"
    endif
    if ($repo2_exists == 1 && $auth2_status != 0) then
        echo "• Fix authentication for sdodlapa"
    endif
endif
