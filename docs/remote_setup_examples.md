# GitHub Remote Setup Examples

## Current Setup
You currently have one remote:
- `origin` → https://github.com/SanjeevaRDodlapati/UAVarPrior.git

## Adding Additional Remotes

### 1. Add Personal Backup Repository
If you have your own GitHub account where you want to backup:
```bash
git remote add personal https://github.com/YOUR_USERNAME/UAVarPrior.git
```

### 2. Add Lab/Organization Repository
If your lab or organization has a GitHub account:
```bash
git remote add lab https://github.com/LAB_ORG_NAME/UAVarPrior.git
```

### 3. Add Collaborator's Fork
If collaborators have forked the repository:
```bash
git remote add collaborator https://github.com/COLLABORATOR_USERNAME/UAVarPrior.git
```

### 4. Add Secondary Account
If you have multiple GitHub accounts:
```bash
git remote add secondary https://github.com/SECOND_ACCOUNT/UAVarPrior.git
```

## Authentication Setup

### For HTTPS (Recommended)
1. Generate a Personal Access Token on GitHub:
   - Go to GitHub Settings → Developer settings → Personal access tokens
   - Generate new token with repo permissions
   - Use token as password when prompted

### For SSH (Alternative)
1. Generate SSH key:
```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
```

2. Add to SSH agent:
```bash
ssh-add ~/.ssh/id_ed25519
```

3. Add public key to GitHub account

4. Use SSH URLs instead:
```bash
git remote add personal git@github.com:YOUR_USERNAME/UAVarPrior.git
```

## Testing Push Access

### Test Individual Remote
```bash
git push personal main
```

### Test All Remotes
```bash
./push_all.csh
```

### Check Remote Status
```bash
git remote -v
git remote show origin
```

## Common Issues and Solutions

### Issue: Permission denied
**Solution**: Check authentication (token/SSH key)

### Issue: Repository doesn't exist
**Solution**: Create repository on GitHub first or check URL

### Issue: Branch doesn't match
**Solution**: 
```bash
git push remote_name main:main
# or
git push remote_name HEAD:main
```

## Quick Setup Commands

Replace `YOUR_USERNAME` with your actual GitHub username:

```bash
# Add your personal backup
git remote add backup https://github.com/YOUR_USERNAME/UAVarPrior.git

# Test the connection
git ls-remote backup

# Push to backup
git push backup main
```

## Git Alias for Easy Multi-Push

The `git pushall` alias is already set up. You can also create additional aliases:

```bash
# Push to specific set of remotes
git config --global alias.pushdev '!git push origin main && git push backup main'

# Push with status for each remote
git config --global alias.pushstatus '!for remote in $(git remote); do echo "Pushing to $remote..."; git push $remote main && echo "✅ $remote" || echo "❌ $remote"; done'
```
