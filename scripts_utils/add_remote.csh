#!/bin/tcsh
# Script to easily add common types of remotes

echo "Add Remote Repository Script"
echo "=============================="
echo ""
echo "Current remotes:"
git remote -v
echo ""

echo "Choose what type of remote to add:"
echo "1) Personal backup repository"
echo "2) Lab/Organization repository" 
echo "3) Collaborator's fork"
echo "4) Custom URL"
echo ""

echo -n "Enter choice (1-4): "
set choice = $<

switch ($choice)
    case 1:
        echo -n "Enter your GitHub username: "
        set username = $<
        git remote add personal https://github.com/$username/UAVarPrior.git
        echo "Added personal remote: https://github.com/$username/UAVarPrior.git"
        breaksw
    case 2:
        echo -n "Enter organization name: "
        set org = $<
        git remote add lab https://github.com/$org/UAVarPrior.git
        echo "Added lab remote: https://github.com/$org/UAVarPrior.git"
        breaksw
    case 3:
        echo -n "Enter collaborator's GitHub username: "
        set collab = $<
        git remote add collaborator https://github.com/$collab/UAVarPrior.git
        echo "Added collaborator remote: https://github.com/$collab/UAVarPrior.git"
        breaksw
    case 4:
        echo -n "Enter remote name: "
        set remote_name = $<
        echo -n "Enter full GitHub URL: "
        set url = $<
        git remote add $remote_name $url
        echo "Added custom remote: $remote_name -> $url"
        breaksw
    default:
        echo "Invalid choice. Exiting."
        exit 1
endsw

echo ""
echo "Updated remotes:"
git remote -v
