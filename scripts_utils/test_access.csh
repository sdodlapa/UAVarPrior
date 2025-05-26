#!/bin/tcsh
# Test if repositories exist and are accessible

echo "🔍 Testing GitHub repository access..."
echo "====================================="
echo ""

echo "Testing primary account (SanjeevaRDodlapati):"
git ls-remote origin HEAD > /dev/null
if ($status == 0) then
    echo "✅ Primary repository accessible"
else
    echo "❌ Primary repository not accessible"
endif

echo ""
echo "Testing secondary account (sdodlapati3):"
git ls-remote secondary HEAD > /dev/null
if ($status == 0) then
    echo "✅ Secondary repository accessible"
else
    echo "❌ Secondary repository not accessible or doesn't exist"
    echo ""
    echo "📝 To fix this:"
    echo "   1. Go to: https://github.com/sdodlapati3"
    echo "   2. Create a new repository named 'UAVarPrior'"
    echo "   3. Don't initialize with README"
    echo "   4. Make it Public"
    echo "   5. Then run: git push secondary main"
endif

echo ""
echo "Current remotes:"
git remote -v
