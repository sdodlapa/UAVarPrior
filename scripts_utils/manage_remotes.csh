#!/bin/tcsh
# Enhanced remote management script

echo "🔧 GitHub Remote Management Tool"
echo "================================="
echo ""

# Show current setup
echo "📍 Current remotes:"
git remote -v
echo ""

# Check if common repositories exist
echo "🔍 Checking for common repository patterns..."
echo ""

set base_url = "https://github.com"
set repo_name = "UAVarPrior"

# Common username patterns to check
set potential_users = ("sdodl001" "SanjeevaRDodlapati" "sdodlapati3")

echo "Checking access to potential repositories:"
foreach user ($potential_users)
    set test_url = "$base_url/$user/$repo_name.git"
    echo -n "Testing $user/$repo_name... "
    
    # Test if repository exists and is accessible
    git ls-remote $test_url >& /dev/null
    if ($status == 0) then
        echo "✅ Accessible"
        
        # Check if it's already added as a remote
        set existing_remote = `git remote -v | grep "$test_url" | awk '{print $1}' | head -1`
        if ("$existing_remote" != "") then
            echo "   Already added as remote: $existing_remote"
        else
            echo -n "   Add as remote? (y/N): "
            set add_choice = $<
            if ("$add_choice" == "y" || "$add_choice" == "Y") then
                echo -n "   Enter remote name (default: $user): "
                set remote_name = $<
                if ("$remote_name" == "") set remote_name = $user
                
                git remote add $remote_name $test_url
                echo "   ✅ Added as remote: $remote_name"
            endif
        endif
    else
        echo "❌ Not accessible or doesn't exist"
    endif
    echo ""
end

echo "📝 Available actions:"
echo "1) Add custom remote"
echo "2) Test push to all remotes"  
echo "3) Show detailed remote info"
echo "4) Remove a remote"
echo "5) Exit"
echo ""

echo -n "Choose action (1-5): "
set action = $<

switch ($action)
    case 1:
        echo -n "Enter GitHub username: "
        set username = $<
        echo -n "Enter remote name (default: $username): "
        set remote_name = $<
        if ("$remote_name" == "") set remote_name = $username
        
        set new_url = "$base_url/$username/$repo_name.git"
        echo "Testing access to $new_url..."
        
        git ls-remote $new_url >& /dev/null
        if ($status == 0) then
            git remote add $remote_name $new_url
            echo "✅ Successfully added remote: $remote_name -> $new_url"
        else
            echo "❌ Cannot access $new_url"
            echo "Make sure the repository exists and you have access."
        endif
        breaksw
        
    case 2:
        echo "🚀 Testing push to all remotes..."
        ./push_all.csh
        breaksw
        
    case 3:
        echo "📊 Detailed remote information:"
        foreach remote (`git remote`)
            echo ""
            echo "Remote: $remote"
            echo "URL: `git remote get-url $remote`"
            git remote show $remote 2>/dev/null || echo "Cannot get detailed info (may need authentication)"
        end
        breaksw
        
    case 4:
        echo "Current remotes:"
        git remote -v
        echo -n "Enter remote name to remove: "
        set remove_remote = $<
        if ("$remove_remote" != "") then
            git remote remove $remove_remote
            echo "✅ Removed remote: $remove_remote"
        endif
        breaksw
        
    case 5:
        echo "👋 Goodbye!"
        exit 0
        breaksw
        
    default:
        echo "Invalid choice."
        breaksw
endsw

echo ""
echo "Final remote configuration:"
git remote -v
