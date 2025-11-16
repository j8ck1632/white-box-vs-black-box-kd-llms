# Setting Up GitHub Repository

## Option 1: Create Repository via GitHub Website (Easiest)

1. Go to https://github.com/new
2. Repository name: `white-box-vs-black-box-kd-llms` (or your preferred name)
3. Choose Public or Private
4. **DO NOT** initialize with README, .gitignore, or license (we already have these)
5. Click "Create repository"

Then run these commands:

```bash
cd "c:\Users\dovak\Documents\white box vs black box in llms"
git remote add origin https://github.com/YOUR_USERNAME/white-box-vs-black-box-kd-llms.git
git branch -M main
git push -u origin main
```

## Option 2: Install GitHub CLI and Create Automatically

1. Install GitHub CLI from: https://cli.github.com/
2. Authenticate: `gh auth login`
3. Run:
   ```bash
   cd "c:\Users\dovak\Documents\white box vs black box in llms"
   gh repo create white-box-vs-black-box-kd-llms --public --source=. --remote=origin --push
   ```

