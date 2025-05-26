# GitHub Authentication Setup for Multiple Accounts

## Method 1: Personal Access Token (Recommended)

### Step 1: Create Personal Access Token for sdodlapati3
1. Go to https://github.com/sdodlapati3
2. Click your profile picture → Settings
3. Scroll down to "Developer settings" (left sidebar)
4. Click "Personal access tokens" → "Tokens (classic)"
5. Click "Generate new token (classic)"
6. Give it a name like "UAVarPrior-push-access"
7. Select scopes:
   - ✅ repo (Full control of private repositories)
   - ✅ workflow (Update GitHub Action workflows)
8. Click "Generate token"
9. **IMPORTANT**: Copy the token immediately (you won't see it again!)

### Step 2: Store Token Securely
```bash
# Create a credential helper to store the token
git config --global credential.helper store

# The token will be stored when you first push
```

### Step 3: Test Authentication
```bash
# First, create the repository on GitHub (if it doesn't exist)
# Go to https://github.com/sdodlapati3 and create a new repository named "UAVarPrior"

# Then test the push
git push secondary main
# When prompted for username: sdodlapati3
# When prompted for password: paste your Personal Access Token
```

## Method 2: SSH Keys for Each Account

### Step 1: Generate SSH Key for Second Account
```bash
# Generate a new SSH key specifically for sdodlapati3
ssh-keygen -t ed25519 -C "sdodl001@odu.edu" -f ~/.ssh/id_ed25519_sdodlapati3

# Start SSH agent
ssh-agent tcsh

# Add the new key
ssh-add ~/.ssh/id_ed25519_sdodlapati3
```

### Step 2: Add SSH Key to GitHub
1. Copy the public key:
```bash
cat ~/.ssh/id_ed25519_sdodlapati3.pub
```
2. Go to https://github.com/sdodlapati3/settings/keys
3. Click "New SSH key"
4. Paste the public key
5. Give it a title like "ODU-Wahab-Server"

### Step 3: Configure SSH for Multiple Accounts
```bash
# Create SSH config file
echo '# GitHub accounts configuration
Host github-main
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_ed25519

Host github-secondary
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_ed25519_sdodlapati3' >> ~/.ssh/config

# Update remote URL to use SSH
git remote set-url secondary git@github-secondary:sdodlapati3/UAVarPrior.git
```

## Method 3: Git Credential Manager (Advanced)

### For managing multiple accounts automatically:
```bash
# Install git credential manager
# (This may require admin access)
git config --global credential.helper manager

# Configure different credentials per remote
git config --local credential.https://github.com/sdodlapati3.username sdodlapati3
```

## Quick Setup Script

Here's what I recommend for the simplest setup:
