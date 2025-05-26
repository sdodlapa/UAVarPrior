#!/bin/csh
echo "🔐 Manual Personal Access Token Setup"
echo "====================================="
echo ""

# Remove existing credentials
if (-f ~/.git-credentials) then
    rm ~/.git-credentials
    echo "🧹 Cleared existing credentials"
endif

# Set up credential helper
git config --global credential.helper store
echo "✅ Configured credential storage"

# Create credentials file manually
echo "📝 Creating credential entries..."

# Note: Replace with actual tokens when you have them
cat > ~/.git-credentials << 'EOF'
https://SanjeevaRDodlapati:TOKEN1@github.com
https://sdodlapati3:TOKEN2@github.com
https://sdodlapa:ghp_XdNR3nYlLnvzPu7wz5kpUCBhZwqFMJ2KF6k4@github.com
EOF

echo "✅ Created credential file with tokens"
echo ""

# Set file permissions
chmod 600 ~/.git-credentials
echo "🔒 Set secure permissions on credential file"

echo ""
echo "Testing authentication for each account:"
echo ""

# Test each account
echo "🔑 Testing SanjeevaRDodlapati..."
git ls-remote https://github.com/SanjeevaRDodlapati/UAVarPrior.git >/dev/null 2>&1
if ($status == 0) then
    echo "✅ SanjeevaRDodlapati: SUCCESS"
else
    echo "❌ SanjeevaRDodlapati: FAILED (need to update TOKEN1)"
endif

echo "🔑 Testing sdodlapati3..."
git ls-remote https://github.com/sdodlapati3/UAVarPrior.git >/dev/null 2>&1
if ($status == 0) then
    echo "✅ sdodlapati3: SUCCESS"
else
    echo "❌ sdodlapati3: FAILED (need to update TOKEN2)"
endif

echo "🔑 Testing sdodlapa..."
git ls-remote https://github.com/sdodlapa/UAVarPrior.git >/dev/null 2>&1
if ($status == 0) then
    echo "✅ sdodlapa: SUCCESS"
else
    echo "❌ sdodlapa: FAILED (check token)"
endif

echo ""
echo "🎯 Manual setup complete!"
echo "💡 Note: Update TOKEN1 and TOKEN2 with actual Personal Access Tokens"
