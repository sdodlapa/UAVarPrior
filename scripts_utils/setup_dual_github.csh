#!/bin/tcsh
# Setup script to add both GitHub accounts to any repository

echo "🔧 GitHub Dual Account Setup"
echo "============================"
echo ""

# Get repository name from current directory or git remote
set repo_name = `basename $PWD`
echo "📂 Current repository: $repo_name"
echo ""

# Check if we're in a git repository
git status >& /dev/null
if ($status != 0) then
    echo "❌ Not in a Git repository. Please run this from a Git repository."
    exit 1
endif

echo "📍 Current remotes:"
git remote -v
echo ""

# Your GitHub accounts
set account1 = "SanjeevaRDodlapati"
set account2 = "sdodlapati3"

# Check and add remotes for both accounts
foreach account ($account1 $account2)
    set remote_url = "https://github.com/$account/$repo_name.git"
    set remote_name = "$account"
    
    # Use shorter name for second account
    if ("$account" == "sdodlapati3") then
        set remote_name = "sdodlapati3"
    else
        set remote_name = "origin"
    endif
    
    # Check if remote already exists
    git remote get-url $remote_name >& /dev/null
    if ($status == 0) then
        echo "✅ Remote '$remote_name' already exists for $account"
    else
        echo "➕ Adding remote '$remote_name' for $account..."
        git remote add $remote_name $remote_url
        
        if ($status == 0) then
            echo "✅ Successfully added $remote_name -> $remote_url"
        else
            echo "❌ Failed to add remote for $account"
        endif
    endif
end

echo ""
echo "📊 Final remote configuration:"
git remote -v
echo ""

# Copy the push scripts to current directory
set script_source = "/home/sdodl001/UAVarPrior"
if (-f "$script_source/push_all.csh") then
    echo "📋 Copying push scripts to current repository..."
    cp "$script_source/push_all.csh" .
    cp "$script_source/quick_push.csh" .
    chmod +x push_all.csh quick_push.csh
    echo "✅ Scripts copied and made executable"
    echo ""
    echo "💡 Usage:"
    echo "  ./push_all.csh     - Push current branch to all remotes"
    echo "  ./quick_push.csh   - Commit changes and push to all remotes"
    echo "  git pushall        - Global alias for pushing to all remotes"
    echo "  git commitpush \"message\" - Global alias for commit + push all"
else
    echo "⚠️  Could not find source scripts in $script_source"
endif

echo ""
echo "🎉 Setup complete! Both GitHub accounts are now configured."
