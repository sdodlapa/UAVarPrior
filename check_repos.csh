#!/bin/csh
echo "🔍 Repository Status Check"
echo "========================="
echo ""

echo "Checking repository existence for all accounts:"
echo ""

# Check SanjeevaRDodlapati repositories
echo "🏢 SanjeevaRDodlapati Account:"
set status1 = `curl -s -o /dev/null -w '%{http_code}' https://github.com/SanjeevaRDodlapati/UAVarPrior`
echo "  UAVarPrior: $status1"
set status2 = `curl -s -o /dev/null -w '%{http_code}' https://github.com/SanjeevaRDodlapati/GenomicLightning`
echo "  GenomicLightning: $status2"
set status3 = `curl -s -o /dev/null -w '%{http_code}' https://github.com/SanjeevaRDodlapati/FuGEP`
echo "  FuGEP: $status3"

echo ""

# Check sdodlapati3 repositories
echo "🏢 sdodlapati3 Account:"
set status4 = `curl -s -o /dev/null -w '%{http_code}' https://github.com/sdodlapati3/UAVarPrior`
echo "  UAVarPrior: $status4"
set status5 = `curl -s -o /dev/null -w '%{http_code}' https://github.com/sdodlapati3/GenomicLightning`
echo "  GenomicLightning: $status5"
set status6 = `curl -s -o /dev/null -w '%{http_code}' https://github.com/sdodlapati3/FuGEP`
echo "  FuGEP: $status6"

echo ""

# Check sdodlapa repositories  
echo "🏢 sdodlapa Account:"
set status7 = `curl -s -o /dev/null -w '%{http_code}' https://github.com/sdodlapa/UAVarPrior`
echo "  UAVarPrior: $status7"
set status8 = `curl -s -o /dev/null -w '%{http_code}' https://github.com/sdodlapa/GenomicLightning`
echo "  GenomicLightning: $status8"
set status9 = `curl -s -o /dev/null -w '%{http_code}' https://github.com/sdodlapa/FuGEP`
echo "  FuGEP: $status9"

echo ""
echo "HTTP Status Codes:"
echo "  200 = Repository exists and is accessible"
echo "  404 = Repository not found or private"
echo "  403 = Access forbidden"

echo ""
echo "Current git remotes for this repository:"
git remote -v
